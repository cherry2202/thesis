from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class ProgressiveLETTER(T5ForConditionalGeneration):
    """T5 recommender with progressive prefix supervision and adaptive latent fusion."""

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.prefix_levels = []
        self.prefix_loss_weights = []
        self.fine_loss_weight = 1.0
        self.prefix_latent_weight = 0.5
        self.temperature = 1.0

        d_model = config.d_model
        vocab_size = config.vocab_size
        self.fine_head = nn.Linear(d_model, vocab_size, bias=False)
        self.prefix_heads = nn.ModuleDict()
        self.prefix_proj = nn.ModuleDict()
        self.fusion_gate = nn.Linear(d_model * 2, d_model, bias=True)
        self.fusion_ln = nn.LayerNorm(d_model)
        self._init_from_lm_head()

    def _init_from_lm_head(self):
        with torch.no_grad():
            self.fine_head.weight.copy_(self.lm_head.weight)
        nn.init.xavier_uniform_(self.fusion_gate.weight)
        nn.init.zeros_(self.fusion_gate.bias)

    def configure_progressive(self, prefix_levels, prefix_loss_weights, fine_loss_weight=1.0, prefix_latent_weight=0.5, temperature=1.0):
        self.prefix_levels = [int(x) for x in prefix_levels]
        self.prefix_loss_weights = [float(x) for x in prefix_loss_weights]
        if len(self.prefix_levels) != len(self.prefix_loss_weights):
            raise ValueError("prefix_levels and prefix_loss_weights must have the same length")
        self.fine_loss_weight = fine_loss_weight
        self.prefix_latent_weight = prefix_latent_weight
        self.temperature = temperature
        self._ensure_prefix_modules()

    def _ensure_prefix_modules(self):
        device = self.shared.weight.device
        dtype = self.shared.weight.dtype
        for level in self.prefix_levels:
            key = str(level)
            if key not in self.prefix_heads:
                head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False).to(device=device, dtype=dtype)
                proj = nn.Linear(self.config.d_model, self.config.d_model, bias=False).to(device=device, dtype=dtype)
                with torch.no_grad():
                    head.weight.copy_(self.lm_head.weight)
                nn.init.xavier_uniform_(proj.weight)
                self.prefix_heads[key] = head
                self.prefix_proj[key] = proj

    def resize_token_embeddings_and_heads(self, new_num_tokens):
        old_fine = self.fine_head.weight.data.clone()
        old_prefix = {k: v.weight.data.clone() for k, v in self.prefix_heads.items()}
        super().resize_token_embeddings(new_num_tokens)
        device = self.shared.weight.device
        dtype = self.shared.weight.dtype
        old_vocab = old_fine.size(0)

        self.fine_head = nn.Linear(self.config.d_model, new_num_tokens, bias=False).to(device=device, dtype=dtype)
        with torch.no_grad():
            self.fine_head.weight[:old_vocab].copy_(old_fine.to(device=device, dtype=dtype))
            if new_num_tokens > old_vocab:
                self.fine_head.weight[old_vocab:].copy_(self.lm_head.weight[old_vocab:].to(device=device, dtype=dtype))

        for key, old_weight in old_prefix.items():
            new_head = nn.Linear(self.config.d_model, new_num_tokens, bias=False).to(device=device, dtype=dtype)
            old_vocab_for_head = old_weight.size(0)
            with torch.no_grad():
                new_head.weight[:old_vocab_for_head].copy_(old_weight.to(device=device, dtype=dtype))
                if new_num_tokens > old_vocab_for_head:
                    new_head.weight[old_vocab_for_head:].copy_(self.lm_head.weight[old_vocab_for_head:].to(device=device, dtype=dtype))
            self.prefix_heads[key] = new_head

        self.config.vocab_size = new_num_tokens
        return self.get_input_embeddings()

    def _masked_ce(self, logits, labels):
        labels = labels.to(logits.device)
        return CrossEntropyLoss(ignore_index=-100)(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

    def _scale_hidden(self, hidden):
        if self.config.tie_word_embeddings:
            return hidden * (self.model_dim ** -0.5)
        return hidden

    def _soft_embedding(self, logits):
        probs = F.softmax(logits / self.temperature, dim=-1)
        return torch.matmul(probs, self.shared.weight)

    def _fuse_prefix_latent(self, hidden, prefix_latent):
        gate = torch.sigmoid(self.fusion_gate(torch.cat([hidden, prefix_latent], dim=-1)))
        return self.fusion_ln(hidden + self.prefix_latent_weight * gate * prefix_latent)

    def forward(self, input_ids=None, attention_mask=None, labels=None, fine_labels=None, prefix_labels=None, **kwargs):
        if fine_labels is None:
            fine_labels = labels
        if fine_labels is None:
            raise ValueError("fine_labels or labels must be provided")
        if prefix_labels is None:
            prefix_labels = {}
        self._ensure_prefix_modules()

        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden_states = encoder_outputs.last_hidden_state
        batch_size = hidden_states.size(0)
        device = hidden_states.device
        tgt_len = fine_labels.size(1)

        start_id = self.config.decoder_start_token_id
        if start_id is None:
            start_id = self.config.pad_token_id if self.config.pad_token_id is not None else 0
        decoder_input = torch.full((batch_size, 1), start_id, dtype=torch.long, device=device)

        decoder_outputs = self.decoder(
            input_ids=decoder_input,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        last_hidden = self._scale_hidden(decoder_outputs.last_hidden_state[:, -1, :])
        past_key_values = decoder_outputs.past_key_values

        fine_logits_steps = []
        prefix_logits_by_level = {str(level): [] for level in self.prefix_levels}

        for step in range(tgt_len):
            prefix_latents = []
            for level in self.prefix_levels:
                key = str(level)
                prefix_logits = self.prefix_heads[key](last_hidden)
                prefix_logits_by_level[key].append(prefix_logits)
                prefix_latents.append(self.prefix_proj[key](self._soft_embedding(prefix_logits)))

            if prefix_latents:
                stacked = torch.stack(prefix_latents, dim=0).mean(dim=0)
                fine_hidden = self._fuse_prefix_latent(last_hidden, stacked)
            else:
                fine_hidden = last_hidden

            fine_logits_steps.append(self.fine_head(fine_hidden))

            if step < tgt_len - 1:
                gold_fine = fine_labels[:, step].clone().to(device)
                gold_fine[gold_fine == -100] = self.config.pad_token_id if self.config.pad_token_id is not None else 0
                decoder_outputs = self.decoder(
                    input_ids=gold_fine.unsqueeze(1),
                    past_key_values=past_key_values,
                    encoder_hidden_states=hidden_states,
                    encoder_attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
                last_hidden = self._scale_hidden(decoder_outputs.last_hidden_state[:, -1, :])
                past_key_values = decoder_outputs.past_key_values

        fine_logits = torch.stack(fine_logits_steps, dim=1)
        fine_loss = self._masked_ce(fine_logits, fine_labels)
        loss = self.fine_loss_weight * fine_loss

        prefix_losses = {}
        for level, weight in zip(self.prefix_levels, self.prefix_loss_weights):
            key = str(level)
            if key not in prefix_labels:
                continue
            level_logits = torch.stack(prefix_logits_by_level[key], dim=1)[:, : prefix_labels[key].size(1), :]
            level_loss = self._masked_ce(level_logits, prefix_labels[key])
            prefix_losses[key] = level_loss
            loss = loss + weight * level_loss

        output = Seq2SeqLMOutput(
            loss=loss,
            logits=fine_logits,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        )
        output.fine_loss = fine_loss
        output.prefix_losses = prefix_losses
        return output
