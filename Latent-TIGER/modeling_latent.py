import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.generation.utils import GenerationMixin
from typing import Optional, Tuple, List

class LATENT(T5ForConditionalGeneration):
    """
    基于 T5 的解码器，支持解码前进行若干步无监督隐式推理（latent reasoning）。
    训练时采用 teacher forcing 一次性计算所有位置的 logits（方案 B）。
    生成时完全兼容 Hugging Face generate 接口，支持 beam search 和前缀约束。
    """
    def __init__(self, config, num_latent_steps=2):
        super().__init__(config)
        self.num_latent_steps = num_latent_steps

        d_model = config.d_model
        # 可学习的初始隐嵌入，用于隐式推理的第一步
        self.latent_init_embedding = nn.Parameter(torch.randn(d_model))

        # 单独的输出头（用于最终预测）
        self.fine_head = nn.Linear(d_model, config.vocab_size, bias=False)
        self._init_fine_head()

        # 为了兼容原 LETTER，保留 model_parallel 属性（但未启用）
        self.model_parallel = False

    def _init_fine_head(self):
        """将 fine_head 初始化为与 lm_head 相同（共享词表）"""
        with torch.no_grad():
            self.fine_head.weight.copy_(self.lm_head.weight)

    def resize_token_embeddings(self, new_num_tokens):
        """
        重写父类方法，确保 fine_head 也同步调整大小。
        """
        old_fine_weight = self.fine_head.weight.data.clone()
        super().resize_token_embeddings(new_num_tokens)
        d_model = self.config.d_model
        device = self.shared.weight.device
        dtype = self.shared.weight.dtype
        self.fine_head = nn.Linear(d_model, new_num_tokens, bias=False).to(device=device, dtype=dtype)
        with torch.no_grad():
            old_vocab = old_fine_weight.size(0)
            self.fine_head.weight[:old_vocab].copy_(old_fine_weight)
            if new_num_tokens > old_vocab:
                self.fine_head.weight[old_vocab:].copy_(self.lm_head.weight[old_vocab:])
        self.config.vocab_size = new_num_tokens
        return self.get_input_embeddings()

    def _masked_ce_loss(self, logits, labels):
        """计算忽略 -100 的交叉熵损失"""
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        labels = labels.to(logits.device)
        return loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

    def _maybe_scale_hidden(self, hidden):
        """如果启用词嵌入绑定，进行缩放"""
        if self.config.tie_word_embeddings:
            hidden = hidden * (self.model_dim ** -0.5)
        return hidden

    def _decoder_prefill(self, encoder_hidden_states, encoder_attention_mask, decoder_input_ids):
        """
        用起始 token 预填充解码器，返回 last_hidden 和 past_key_values
        """
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        last_hidden = decoder_outputs.last_hidden_state[:, -1, :]  # [B, D]
        last_hidden = self._maybe_scale_hidden(last_hidden)
        return last_hidden, decoder_outputs.past_key_values

    def _decoder_step(self, encoder_hidden_states, encoder_attention_mask,
                      past_key_values, input_ids=None, inputs_embeds=None):
        """
        解码器单步前向，返回 last_hidden 和更新后的 past_key_values
        """
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
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

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        labels=None,
        past_key_values=None,          # 用于生成模式
        inputs_embeds=None,            # 用于生成模式
        head_mask=None,
        decoder_head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        num_latent_steps=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ------ 生成模式 (labels is None) ------
        # if labels is None:
        #     # 编码器
        #     if encoder_outputs is None:
        #         encoder_outputs = self.encoder(
        #             input_ids=input_ids,
        #             attention_mask=attention_mask,
        #             head_mask=head_mask,
        #             output_attentions=output_attentions,
        #             output_hidden_states=output_hidden_states,
        #             return_dict=True,
        #         )
        #     hidden_states = encoder_outputs.last_hidden_state
        #     # 处理 batch size
        #     if input_ids is not None:
        #         batch_size = input_ids.size(0)
        #         device = input_ids.device
        #     else:
        #         batch_size = inputs_embeds.size(0)
        #         device = inputs_embeds.device

        #     # 首次生成（past_key_values is None）：执行预填充和 latent
        #     if past_key_values is None:
        #         decoder_start_token_id = self.config.decoder_start_token_id
        #         if decoder_start_token_id is None:
        #             decoder_start_token_id = self.config.pad_token_id
        #         if decoder_start_token_id is None:
        #             decoder_start_token_id = 0
        #         start_ids = torch.full((batch_size, 1), decoder_start_token_id, dtype=torch.long, device=device)
        #         # 预填充
        #         last_hidden, past_key_values = self._decoder_prefill(
        #             encoder_hidden_states=hidden_states,
        #             encoder_attention_mask=attention_mask,
        #             decoder_input_ids=start_ids,
        #         )
        #         # latent 步
        #         steps = num_latent_steps if num_latent_steps is not None else self.num_latent_steps
        #         if steps > 0:
        #             latent_emb = self.latent_init_embedding.unsqueeze(0).expand(batch_size, -1)
        #             for _ in range(steps):
        #                 latent_emb, past_key_values = self._decoder_step(
        #                     encoder_hidden_states=hidden_states,
        #                     encoder_attention_mask=attention_mask,
        #                     past_key_values=past_key_values,
        #                     input_ids=None,
        #                     inputs_embeds=latent_emb.unsqueeze(1),
        #                 )
        #         # 用预填充最后一步的 hidden 计算第一个 token 的 logits
        #         # 注意：预填充的 last_hidden 对应 start_token 的输出，用于预测第一个真正的 token
        #         logits = self.fine_head(last_hidden)  # [B, V]
        #         if not return_dict:
        #             return (logits, past_key_values, encoder_outputs)
        #         return Seq2SeqLMOutput(
        #             logits=logits,
        #             past_key_values=past_key_values,
        #             encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        #         )
        #     else:
        #         # 后续生成步骤：输入是当前 token 的 embeddings，已有 past_key_values
        #         if input_ids is not None:
        #             inputs_embeds = self.shared(input_ids)
        #         # 单步解码
        #         last_hidden, new_past = self._decoder_step(
        #             encoder_hidden_states=hidden_states,
        #             encoder_attention_mask=attention_mask,
        #             past_key_values=past_key_values,
        #             input_ids=None,
        #             inputs_embeds=inputs_embeds,
        #         )
        #         logits = self.fine_head(last_hidden)  # [B, V]
        #         if not return_dict:
        #             return (logits, new_past, encoder_outputs)
        #         return Seq2SeqLMOutput(
        #             logits=logits,
        #             past_key_values=new_past,
        #             encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        #         )
        # ------ 生成模式 (labels is None) ------
        if labels is None:
            # 编码器
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
        
            # 首次生成
            if past_key_values is None:
                decoder_start_token_id = self.config.decoder_start_token_id
                if decoder_start_token_id is None:
                    decoder_start_token_id = self.config.pad_token_id
                if decoder_start_token_id is None:
                    decoder_start_token_id = 0
                start_ids = torch.full((batch_size, 1), decoder_start_token_id, dtype=torch.long, device=device)
                last_hidden, past_key_values = self._decoder_prefill(
                    encoder_hidden_states=hidden_states,
                    encoder_attention_mask=attention_mask,
                    decoder_input_ids=start_ids,
                )
                steps = num_latent_steps if num_latent_steps is not None else self.num_latent_steps
                if steps > 0:
                    latent_emb = self.latent_init_embedding.unsqueeze(0).expand(batch_size, -1)
                    for _ in range(steps):
                        latent_emb, past_key_values = self._decoder_step(
                            encoder_hidden_states=hidden_states,
                            encoder_attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            input_ids=None,
                            inputs_embeds=latent_emb.unsqueeze(1),
                        )
                logits = self.fine_head(last_hidden)          # [B, V]
                logits = logits.unsqueeze(1)                  # [B, 1, V]   <-- 增加维度
                if not return_dict:
                    return (logits, past_key_values, encoder_outputs)
                return Seq2SeqLMOutput(
                    logits=logits,
                    past_key_values=past_key_values,
                    encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                )
            else:
                # 后续生成步骤
                if input_ids is not None:
                    inputs_embeds = self.shared(input_ids)
                last_hidden, new_past = self._decoder_step(
                    encoder_hidden_states=hidden_states,
                    encoder_attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    input_ids=None,
                    inputs_embeds=inputs_embeds,
                )
                logits = self.fine_head(last_hidden)          # [B, V]
                logits = logits.unsqueeze(1)                  # [B, 1, V]   <-- 增加维度
                if not return_dict:
                    return (logits, new_past, encoder_outputs)
                return Seq2SeqLMOutput(
                    logits=logits,
                    past_key_values=new_past,
                    encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                )

        # ------ 训练模式 (labels provided) ------
        # 编码器
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

        # 处理 decoder_start_token_id
        decoder_start_token_id = self.config.decoder_start_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = self.config.pad_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = 0

        start_ids = torch.full(
            (batch_size, 1),
            fill_value=decoder_start_token_id,
            dtype=torch.long,
            device=device
        )

        # ---- 预填充 ----
        last_hidden, past_key_values = self._decoder_prefill(
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            decoder_input_ids=start_ids,
        )

        # ---- 隐式推理步 ----
        latent_steps = num_latent_steps if num_latent_steps is not None else self.num_latent_steps
        if latent_steps > 0:
            latent_emb = self.latent_init_embedding.unsqueeze(0).expand(batch_size, -1)
            for _ in range(latent_steps):
                latent_emb, past_key_values = self._decoder_step(
                    encoder_hidden_states=hidden_states,
                    encoder_attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    input_ids=None,
                    inputs_embeds=latent_emb.unsqueeze(1),
                )

        # ---- 方案 B：一次性输入所有目标 token（teacher forcing） ----
        labels_for_input = labels.clone()
        labels_for_input[labels_for_input == -100] = self.config.pad_token_id
        inputs_embeds = self.shared(labels_for_input)  # [B, T, D]

        decoder_outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        hidden_states_all = decoder_outputs.last_hidden_state
        # 即使 latent_steps=0，也让 latent_init_embedding 参与计算，避免 DDP 未使用参数错误
        hidden_states_all = hidden_states_all + self.latent_init_embedding * 0.0
        logits_all = self.fine_head(hidden_states_all)  # [B, T, V]

        # 损失：logits[:, :-1] 与 labels[:, 1:] 对齐
        loss = self._masked_ce_loss(logits_all[:, :-1, :], labels[:, 1:])

        if not return_dict:
            return (loss, logits_all)

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits_all,
            past_key_values=None,
            decoder_hidden_states=None,
            decoder_attentions=None,
            cross_attentions=None,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    # ---------- 生成兼容性方法 ----------
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None,
                                      use_cache=None, encoder_outputs=None, **kwargs):
        """
        为 generate 准备输入，确保首次调用时 past_key_values 为 None，
        我们的 forward 会处理预填充和 latent。
        """
        # 如果 past_key_values 为 None，则首次调用，我们保留 input_ids（但会被 forward 中的 start token 覆盖？）
        # 实际上在首次生成时，父类会传入 decoder_start_token_id 作为 input_ids，
        # 但我们的 forward 在 past_key_values=None 时会忽略 input_ids 而使用自己的 start_ids。
        # 为了避免冲突，我们不移除 input_ids，但 forward 会检查 past_key_values 并覆盖。
        # 注意：我们还需要传递 num_latent_steps 等额外参数。
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            **kwargs,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        """
        重排缓存以支持 beam search。
        T5 的 past_key_values 是元组，每个元素是 (key, value) 对，共 num_layers 层。
        我们需要对每个层的 key, value 都进行索引重排。
        """
        # 如果 past_key_values 是 None，直接返回
        if past_key_values is None:
            return None
        # 标准 T5 的缓存格式：tuple of tuple of tensors，每层有 (key, value)
        # 需要确保每层每个张量的 batch 维度被重排
        reordered_past = []
        for layer_past in past_key_values:
            # layer_past 是 (key, value) 元组
            reordered_layer = tuple(
                past_state.index_select(0, beam_idx) for past_state in layer_past
            )
            reordered_past.append(reordered_layer)
        return tuple(reordered_past)