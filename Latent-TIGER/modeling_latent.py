import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from typing import Optional


class LATENT(T5ForConditionalGeneration):
    """
    基于 T5 的解码器，支持解码前进行若干步无监督隐式推理（latent reasoning）。
    训练时：执行 latent 步骤，拼接标准decoder_start_token + latent嵌入作为解码器前两位输入，
            然后 teacher forcing 后续 token。
    生成时：先用decoder_start_token初始化KV缓存，第二位输入latent推演特征，保持序列格式与训练完全一致
    支持 num_latent_steps=0（退化为标准原生T5，无额外输出头、无多余正则梯度干扰）
    """
    def __init__(self, config, num_latent_steps=2):
        super().__init__(config)
        self.num_latent_steps = num_latent_steps
        self.model_dim = config.d_model

        d_model = config.d_model
        self.latent_init_embedding = nn.Parameter(torch.randn(d_model))
        self.model_parallel = False

    def resize_token_embeddings(self, new_num_tokens):
        return super().resize_token_embeddings(new_num_tokens)

    def _masked_ce_loss(self, logits, labels):
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        labels = labels.to(logits.device)
        return loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

    def _maybe_scale_hidden(self, hidden):
        if self.config.tie_word_embeddings:
            hidden = hidden * (self.model_dim ** -0.5)
        return hidden

    def _decoder_prefill(self, encoder_hidden_states, encoder_attention_mask, decoder_input_ids):
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        last_hidden = decoder_outputs.last_hidden_state[:, -1, :]
        last_hidden = self._maybe_scale_hidden(last_hidden)
        return last_hidden, decoder_outputs.past_key_values

    def _decoder_step(self, encoder_hidden_states, encoder_attention_mask,
                      past_key_values, input_ids=None, inputs_embeds=None, use_cache=True):
        assert not (input_ids is not None and inputs_embeds is not None), \
            "Cannot pass both input_ids and inputs_embeds to decoder"
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        last_hidden = decoder_outputs.last_hidden_state[:, -1, :]
        last_hidden = self._maybe_scale_hidden(last_hidden)
        new_past = decoder_outputs.past_key_values if use_cache else None
        return last_hidden, new_past

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        labels=None,
        past_key_values=None,
        inputs_embeds=None,
        head_mask=None,
        decoder_head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        num_latent_steps=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        latent_steps = num_latent_steps if num_latent_steps is not None else self.num_latent_steps
        assert isinstance(latent_steps, int) and latent_steps >= 0, "num_latent_steps must be non-negative integer"

        # ========== 推理生成模式 (labels is None) ==========
        if labels is None:
            if encoder_outputs is None:
                encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                )
            hidden_states = encoder_outputs.last_hidden_state
            if input_ids is not None:
                batch_size = input_ids.size(0)
                device = input_ids.device
            else:
                batch_size = inputs_embeds.size(0)
                device = inputs_embeds.device

            if past_key_values is None:
                # 1. 多步latent推演
                latent_emb = self.latent_init_embedding.unsqueeze(0).expand(batch_size, -1)
                for _ in range(latent_steps):
                    latent_emb, _ = self._decoder_step(
                        encoder_hidden_states=hidden_states,
                        encoder_attention_mask=attention_mask,
                        past_key_values=None,
                        inputs_embeds=latent_emb.unsqueeze(1),
                        use_cache=False
                    )
                # 2. 先用标准start token生成第一层KV缓存（序列第0位）
                start_ids = torch.full((batch_size, 1), self.config.decoder_start_token_id, dtype=torch.long, device=device)
                _, past_key_values = self._decoder_prefill(
                    encoder_hidden_states=hidden_states,
                    encoder_attention_mask=attention_mask,
                    decoder_input_ids=start_ids
                )
                # 3. 第二步输入latent特征（序列第1位），对齐训练拼接格式 [start, latent]
                last_hidden, past_key_values = self._decoder_step(
                    encoder_hidden_states=hidden_states,
                    encoder_attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=latent_emb.unsqueeze(1),
                    use_cache=True
                )
                logits = self.lm_head(last_hidden).unsqueeze(1)
            else:
                if input_ids is not None:
                    inputs_embeds = self.shared(input_ids)
                last_hidden, new_past = self._decoder_step(
                    encoder_hidden_states=hidden_states,
                    encoder_attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=True
                )
                past_key_values = new_past
                logits = self.lm_head(last_hidden).unsqueeze(1)

            if not return_dict:
                return (logits, past_key_values, encoder_outputs)
            return Seq2SeqLMOutput(
                logits=logits,
                past_key_values=past_key_values,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            )

        # ========== 训练模式 (labels provided) ==========
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        hidden_states = encoder_outputs.last_hidden_state
        batch_size, tgt_len = labels.size(0), labels.size(1)
        device = hidden_states.device

        decoder_start_token_id = self.config.decoder_start_token_id or self.config.pad_token_id or 0
        start_ids = torch.full((batch_size, 1), decoder_start_token_id, dtype=torch.long, device=device)
        start_embeds = self.shared(start_ids)

        if latent_steps > 0:
            latent_emb = self.latent_init_embedding.unsqueeze(0).expand(batch_size, -1)
            for _ in range(latent_steps):
                latent_emb, _ = self._decoder_step(
                    encoder_hidden_states=hidden_states,
                    encoder_attention_mask=attention_mask,
                    past_key_values=None,
                    inputs_embeds=latent_emb.unsqueeze(1),
                    use_cache=False
                )
            latent_1 = latent_emb.unsqueeze(1)
            # 【核心修复】训练输入拼接：[start_token, latent] 与推理序列长度对齐
            first_two_emb = torch.cat([start_embeds, latent_1], dim=1)
        else:
            # latent=0时退化为原生T5，仅保留start token
            first_two_emb = start_embeds

        labels_for_input = labels.clone()
        labels_for_input[labels_for_input == -100] = self.config.pad_token_id
        labels_embeds = self.shared(labels_for_input)
        rest_emb = labels_embeds[:, :-1, :]
        # 完整输入序列：[start, latent] + item序列，与推理完全匹配
        inputs_embeds = torch.cat([first_two_emb, rest_emb], dim=1)

        decoder_outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        hidden_states_all = decoder_outputs.last_hidden_state
        logits_all = self.lm_head(hidden_states_all)
        # 前1位是start_token对应的输出，无对应标签，直接舍弃，后面和labels对齐
        logits_align = logits_all[:, 1:, :]
        loss = self._masked_ce_loss(logits_align, labels)

        # loss = self._masked_ce_loss(logits_all, labels)

        if not return_dict:
            return (loss, logits_all, None, encoder_outputs)

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits_all,
            past_key_values=None,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None,
                                      use_cache=None, encoder_outputs=None, **kwargs):
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "num_latent_steps": kwargs.get("num_latent_steps", self.num_latent_steps),
            **kwargs,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        if past_key_values is None:
            return None
        if hasattr(past_key_values, "reorder_cache"):
            return past_key_values.reorder_cache(beam_idx)
        reordered_past = []
        for layer_past in past_key_values:
            reordered_layer = []
            for past_state in layer_past:
                if past_state is not None:
                    reordered_layer.append(past_state.index_select(0, beam_idx))
                else:
                    reordered_layer.append(None)
            reordered_past.append(tuple(reordered_layer))
        return tuple(reordered_past)