from typing import Any, Mapping
import torch
from torch import nn, Tensor
from dataclasses import dataclass
import math
from rotary_embedding_torch import RotaryEmbedding
from flash_attn import flash_attn_varlen_func


@dataclass
class Config:
    vocab_size: int
    num_embeddings: int
    n_layers: int
    n_heads: int
    hidden_size: int
    seq_length: int
    pad_token_id: int
    attn_dropout: float = 0.
    hidden_dropout: float = 0.
    seq_dim_for_attn: int = -2
    nhead_dim_for_attn: int = -3


class Attention(nn.Module):
    def __init__(self, config: Config, rope_fn=None):
        super().__init__()
        self.config = config
        self.rope_fn = rope_fn
        self.query_key_value = nn.Linear(
            self.config.hidden_size, self.config.hidden_size * 3)
        self.dense = nn.Linear(self.config.hidden_size,
                               self.config.hidden_size)

        self.transpose_seq_and_nhead = lambda x: torch.transpose(
            x, config.seq_dim_for_attn, config.nhead_dim_for_attn)

    def _causal_attn(self, Q: Tensor, K: Tensor, V: Tensor, dropout_p=0.):
        qlen, klen = Q.size(-2), K.size(-2)
        if qlen == klen:
            attn_mask = torch.ones(Q.size(0), K.size(
                0), device=Q.device).tril(diagonal=0)
        elif qlen < klen:
            offset = klen - qlen
            attn_mask = torch.zeros(1, klen, device=Q.device)
            attn_mask[:offset + 1] = True
        else:
            raise RuntimeError("Not implemented")

        min_mask_value = torch.finfo(Q.dtype).min

        scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
        scores = torch.masked_fill(scores, not attn_mask if attn_mask.dtype == torch.bool else attn_mask == 0, min_mask_value)
        attn_weight = torch.softmax(scores, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, self.training)
        return attn_weight @ V

    def forward(self, x, prefix_kv=None):
        # get qkv shape: [batch_size, seq_length, hidden_size * 3]
        qkv = self.query_key_value(x)
        head_size = self.config.hidden_size // self.config.n_heads

        # get qkv shape: [batch_size, seq_length, n_heads, head_size * 3]
        qkv = torch.reshape(qkv, list(qkv.shape[:-1]) + [-1, head_size * 3])

        # get qkv shape: [batch_size, n_heads, seq_length, head_size * 3]
        qkv = self.transpose_seq_and_nhead(qkv)

        # get q/k/v shape: [batch_size, n_heads, seq_length, head_size * 3]
        q, k, v = torch.split(qkv, head_size, -1)

        offset = 0
        if prefix_kv is not None:
            pk, pv = prefix_kv
            offset = pk.size(offset)
            k = torch.cat((pk, k), dim=self.config.seq_dim_for_attn)
            v = torch.cat((pv, v), dim=self.config.seq_dim_for_attn)
            prefix_kv = torch.stack((k, v))

        if self.rope_fn is not None:
            q = self.rope_fn(
                q, seq_dim=self.config.seq_dim_for_attn, offset=offset).type_as(v)
            k = self.rope_fn(
                k, seq_dim=self.config.seq_dim_for_attn).type_as(v)

        # qlen = q.size(self.config.seq_dim_for_attn)
        # klen = k.size(self.config.seq_dim_for_attn)
        # flash_attn_varlen_func(
        #     q,
        #     k,
        #     v,
        #     cu_seqlens_q=qlen,
        #     cu_seqlens_k=klen,
        #     max_seqlen_q=self.config.seq_length,
        #     max_seqlen_k=self.config.seq_length,
        #     dropout_p=self.config.attn_dropout,
        #     causal=True
        # )

        attn_out = self._causal_attn(q, k, v, self.config.attn_dropout)
        attn_out = self.transpose_seq_and_nhead(attn_out)
        attn_out = attn_out.reshape_as(x)
        return self.dense(attn_out), prefix_kv


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size, config.hidden_size * 4)
        self.dense_4h_to_h = nn.Linear(
            config.hidden_size * 4, config.hidden_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.dense_h_to_4h(x)
        x = self.dense_4h_to_h(self.gelu(x))
        return x


class TransformerLayer(nn.Module):
    def __init__(self, config: Config, rope_fn=None):
        super().__init__()
        self.config = config
        self.self_attention = Attention(config, rope_fn)
        self.mlp = MLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self, x, prefix_kv=None):
        ipt_ln_out = self.input_layernorm(x)
        attn_out, prefix_kv = self.self_attention(ipt_ln_out, prefix_kv)
        x = x + torch.dropout(attn_out,
                              self.config.hidden_dropout, self.training)

        post_ln_out = self.post_attention_layernorm(x)
        mlp_out = self.mlp(post_ln_out)
        x = x + torch.dropout(mlp_out,
                              self.config.hidden_dropout, self.training)
        return x, prefix_kv


class Encoder(nn.Module):
    def __init__(self, config: Config, rope_fn):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerLayer(config, rope_fn)] * config.n_layers)
        self.final_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self, x, prefix_kv_list=None):
        if prefix_kv_list is None:
            prefix_kv_list = [None] * len(self.layers)

        next_prefix_kv_list = []
        for layer, prefix_kv in zip(self.layers, prefix_kv_list):
            x, next_prefix_kv = layer(x, prefix_kv)
            next_prefix_kv_list.append(next_prefix_kv)

        fnl_ln_out = self.final_layernorm(x)
        return fnl_ln_out, next_prefix_kv_list


class Embedding(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.num_embeddings, config.hidden_size)

    def forward(self, x):
        return self.word_embeddings(x)


class SFLLM(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.rope_emb = RotaryEmbedding(config.hidden_size // config.n_heads)
        self.embedding = Embedding(config)
        self.encoder = Encoder(config, self.rope_emb.rotate_queries_or_keys)

    def forward(self, x, prefix_kv_list=None):
        x = self.embedding(x)
        x, next_prefix_kv_list = self.encoder(x, prefix_kv_list)
        x = x @ self.embedding.word_embeddings.weight.t()
        return x, next_prefix_kv_list

    def load_megatron_checkpoint(self, state_dict: Mapping[str, Any], strict: bool = True):
        res_emb = self.embedding.word_embeddings.load_state_dict(
            state_dict['embedding']['word_embeddings'], strict)
        res_enc = self.encoder.load_state_dict(state_dict['encoder'], strict)
        return res_emb, res_enc

    def _sample(
        self,
        last_token_mat: Tensor,
        top_k: int,
        top_p: float,
        temperature: float = 1,
    ):
        min_mask_value = torch.finfo(last_token_mat.dtype).min
        assert last_token_mat.ndim == 2
        if top_k == 1:
            assert top_p == 0.0, 'cannot set both greedy and top-p samplings.'
            samples = torch.argmax(last_token_mat, dim=-1)
        else:
            last_token_mat = last_token_mat.clone()
            if temperature != 1.0:
                last_token_mat.div_(temperature)
            if top_k > 1:
                assert top_p == 0.0, 'cannot set both top-k and top-p samplings.'
                assert top_k <= last_token_mat.size(
                    1), 'top-k is larger than logit size.'
                assert top_k < self.config.vocab_size, 'top-k is larger than vocab size.'
                filter_ = last_token_mat < torch.topk(
                    last_token_mat, top_k).values[..., -1, None]
                last_token_mat.masked_fill_(filter_, min_mask_value)
            elif top_p > 0.:
                assert top_p <= 1.0, 'top-p should be in (0, 1].'
                sorted_logits, sorted_indices = torch.sort(
                    last_token_mat, descending=True)
                cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                filter_ = cumulative_probs > top_p
                filter_[:, 1:] = filter_[:, :-1].clone()
                filter_[..., 0] = 0
                filter_ = filter_.scatter(1, sorted_indices, filter_)
                last_token_mat.masked_fill_(filter_, min_mask_value)
            probs = last_token_mat.softmax(dim=-1)
            samples = torch.multinomial(probs, num_samples=1).view(-1)

        samples = torch.clamp(samples, min=0, max=(self.config.vocab_size - 1))
        return samples

    def generate(
        self,
        context: Tensor,
        stop_token_id: int,
        top_k=10,
        top_p=0.0,
        top_p_decay=0.0,
        top_p_bound=0.0,
        temperature: float = 1.,
        min_length: int = 10,
        max_length: int = 2048,
        seq_dim: int = 1
    ):
        assert context.size(0) == 1
        assert context.size(
            seq_dim) + min_length <= self.config.seq_length, "Min length can't be longer than min context length."
        assert context.size(
            seq_dim) + max_length <= self.config.seq_length, "Max length can't be longer than max context length."

        prefix_kv_list = None
        input_ids = context
        output_ids = None
        for i in range(max_length):
            logits, prefix_kv_list = self.forward(input_ids, prefix_kv_list)
            last_token_mat = logits[:, -1, :]
            last_token_ids = self._sample(
                last_token_mat, top_k, top_p, temperature)
            if top_p > 0.0 and top_p_decay > 0.0:
                top_p = top_p * top_p_decay
                if top_p_bound > 0.0:
                    top_p = max(top_p, top_p_bound)

            assert last_token_ids.shape == (1,)
            if last_token_ids[0] == stop_token_id:
                break
            
            input_ids = last_token_ids.unsqueeze(0)
            if output_ids is None:
                output_ids = last_token_ids
            else:
                output_ids = torch.cat((output_ids, last_token_ids), dim=-1)

        return output_ids
