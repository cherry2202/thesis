from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import (
    T5Stack, T5Block, T5LayerNorm, T5LayerSelfAttention, T5LayerFF, T5LayerCrossAttention,
    T5PreTrainedModel, T5ForConditionalGeneration
)
import torch
from torch import nn
import copy
import torch
import torch.nn as nn
import json
import torch.nn.functional as F
import numpy as np
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import ModelOutput, BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, Seq2SeqModelOutput
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import logging
from transformers import BeamScorer, BeamSearchScorer
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


class LETTER(T5ForConditionalGeneration):

    def __init__(self, config: T5Config, args):

        super().__init__(config)

        # You can add parameters out here.
        # self.temperature = 1.0
        self.temperature = args.temperature
        # =================新增========================
        self.alpha = args.alpha
        # self.alpha = 0.8
        self.coarse_lm_head = nn.Linear(config.d_model, config.vocab_size)

    def set_hyper(self,temperature):
        self.temperature = temperature


    def ranking_loss(self, lm_logits, labels):
        if labels is None:
            # loss = 0.0
            return torch.tensor(0.0, device=lm_logits.device)
        if labels is not None:
            t_logits = lm_logits/self.temperature
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(t_logits.view(-1, t_logits.size(-1)), labels.view(-1))
        return loss

    # ==============新增：粗粒度损失==========================
    # def total_loss(self, lm_logits, labels, decoder_input_ids):
    #     loss = self.ranking_loss(lm_logits, labels)             
    #     return loss
    def total_loss(self, lm_logits, coarse_lm_logits, labels, coarse_labels, decoder_input_ids):
        # print("coarse_labels exists:", coarse_labels is not None)
        # if coarse_labels is not None:
        #     print("coarse_labels example:", coarse_labels[0])
        loss_fine = self.ranking_loss(lm_logits=lm_logits, labels=labels)
        loss_coarse = self.ranking_loss(lm_logits=coarse_lm_logits, labels=coarse_labels)
        # print(f"✅ fine={loss_fine:.4f} | coarse={loss_coarse:.4f}")
         # 🔥 只在训练时、随机 5% 的概率打印，不会刷屏
        if self.training and torch.rand(1).item() < 0.05:
            print(f"[TRAIN] fine_loss = {loss_fine.item():6.4f} | coarse_loss = {loss_coarse.item():6.4f}")
        total_loss = self.alpha * loss_fine + (1-self.alpha) * loss_coarse
        return total_loss, loss_fine, loss_coarse
    
    
    def forward(
        self,
        input_ids=None,
        whole_word_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        cross_attn_head_mask = None,
        past_key_values=None,
        use_cache=None,
        labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
        decoder_head_mask = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        reduce_loss=False,
        coarse_labels=None, # =============新增：粗粒度标签===============
        return_hidden_state=False,

        **kwargs,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)
        
        # 粗标签如果存在，也用同一个 decoder 输入
        if coarse_labels is not None:
            coarse_decoder_input_ids = self._shift_right(coarse_labels)
        else:
            coarse_decoder_input_ids = None

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,

            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)
        # ====================新增=========================
        coarse_logits = None
        if coarse_labels is not None:
            coarse_decoder_outputs = self.decoder(
                input_ids=coarse_decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=return_dict,
            )
            coarse_sequence_output = coarse_decoder_outputs[0]
            coarse_logits = self.coarse_lm_head(coarse_sequence_output)
        
        # ------------------------------------------
        # Loss Computing!
        loss = None
        # ==================新增：粗粒度损失=====================
        # loss = self.total_loss(lm_logits, labels, decoder_input_ids)
        if labels is not None :
            # L = decoder_input_ids.size(1)

            # labels = labels[:, :L].contiguous()
            # middle_labels = middle_labels[:, :L].contiguous()
            # coarse_labels = coarse_labels[:, :L].contiguous()

            loss, _, _ = self.total_loss(
                lm_logits=lm_logits, 
                coarse_lm_logits=coarse_logits, 
                labels=labels, 
                coarse_labels=coarse_labels, 
                decoder_input_ids=decoder_input_ids
                )
            
        # ------------------------------------------

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits, # ==============推理只用这个，不加辅助任务=================
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
