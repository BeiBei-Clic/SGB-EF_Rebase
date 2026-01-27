from typing import Tuple

import pysnooper
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionBlock(nn.Module):
    """交叉注意力块：序列作为 Q，条件作为 K/V"""
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 序列嵌入 (seq_len, batch_size, hidden_dim)
            condition: 条件嵌入 (num_features, batch_size, hidden_dim)

        Returns:
            输出嵌入 (seq_len, batch_size, hidden_dim)
        """
        # Q: 序列, K/V: 条件
        seq_len, batch_size, _ = x.shape
        num_features = condition.shape[0]

        Q = self.q_proj(x)  # (seq_len, batch, hidden)
        K = self.k_proj(condition)  # (num_features, batch, hidden)
        V = self.v_proj(condition)  # (num_features, batch, hidden)

        # 重塑为多头注意力格式
        head_dim = self.hidden_dim // self.num_heads
        Q = Q.view(seq_len, batch_size, self.num_heads, head_dim).transpose(1, 2)  # (seq, heads, batch, head_dim)
        K = K.view(num_features, batch_size, self.num_heads, head_dim).transpose(1, 2)  # (features, heads, batch, head_dim)
        V = V.view(num_features, batch_size, self.num_heads, head_dim).transpose(1, 2)  # (features, heads, batch, head_dim)

        # 调整为 (heads, batch, seq/features, head_dim) 以便矩阵乘法
        Q = Q.transpose(0, 1)  # (heads, batch, seq, head_dim)
        K = K.transpose(0, 1).transpose(2, 3)  # (heads, batch, head_dim, features)
        V = V.transpose(0, 1)  # (heads, batch, features, head_dim)

        # 注意力计算
        attn = (Q @ K) / (head_dim ** 0.5)  # (heads, batch, seq, features)
        attn = F.softmax(attn, dim=-1)

        out = attn @ V  # (heads, batch, seq, head_dim)
        out = out.transpose(1, 2).contiguous()  # (seq, batch, heads, head_dim)
        out = out.view(seq_len, batch_size, self.hidden_dim)  # (seq, batch, hidden)

        # 残差连接 + FFN
        x = self.norm1(x + self.out_proj(out))
        x = self.norm2(x + self.ffn(x))
        return x


class SinusoidalTimeEmbedding(nn.Module):
    """
    Simple sinusoidal time embedding for Transformer model
    """
    def __init__(self, hidden_dim: int):
        super(SinusoidalTimeEmbedding, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # (batch, 1) -> (batch, hidden_dim)
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # Make it (batch_size, 1)

        half_dim = self.hidden_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t * emb.unsqueeze(0)  # Broadcasting: (batch_size, 1) * (1, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # Handle odd hidden_dim
        if self.hidden_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

        return emb  # (batch_size, hidden_dim)

class EditFlowsTransformer(nn.Module):
    """
    Small vanilla Transformer model for edit flows with padding support.
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads = 8,
        max_seq_len=512,
        bos_token_id=128,
        pad_token_id=129,
        num_cond_features=1,
    ):
        super(EditFlowsTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.num_cond_features = num_cond_features
        assert bos_token_id < vocab_size, "bos_token_id must be less than vocab_size"
        assert pad_token_id < vocab_size, "pad_token_id must be less than vocab_size"

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)     # Token embeddings
        self.time_embedding = nn.Sequential(                            # Time embeddings
            SinusoidalTimeEmbedding(hidden_dim=hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)      # Positional embeddings
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
                                        dim_feedforward=hidden_dim * 4,
                                        dropout=0.1, activation='gelu',
                                        batch_first=False)
            for _ in range(num_layers)
        ])

        # 交叉注意力层（每层一个）
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        # 条件投影层：将条件投影到 hidden_dim
        self.cond_proj = nn.Linear(hidden_dim, hidden_dim)
        self.final_layer_norm = nn.LayerNorm(hidden_dim)

        self.rates_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),  # Output 3 rates (insert, substitute, delete)
        )
        self.ins_logits_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, vocab_size),  # Output vocab_size insert probabilities
        )
        self.sub_logits_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, vocab_size),  # Output vocab_size substitute probabilities
        )
        self._init_weights()  # Initialize weights

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, std=0.02)

    @pysnooper.snoop('logs/debug.log', watch=[
        'torch.isnan(x).any()',
        'torch.isnan(ins_logits).any()',
        'torch.isnan(sub_logits).any()',
        'torch.isnan(rates).any()',
        'torch.isnan(ins_probs).any()',
        'torch.isnan(sub_probs).any()',
        'x.min()',
        'x.max()',
        'ins_logits.min()',
        'ins_logits.max()',
    ])
    def forward(self, tokens: torch.Tensor,  # (batch, x_seq_len)
                time_step: torch.Tensor,  # (batch, 1)
                padding_mask: torch.Tensor,  # (batch, x_seq_len)
                condition: torch.Tensor,  # (batch, num_features, hidden_dim)
                ) -> Tuple[
            torch.Tensor,  # Rates (batch, x_seq_len, 3)
            torch.Tensor,  # Insert probabilities (batch, x_seq_len, vocab_size)
            torch.Tensor,  # Substitute probabilities (batch, x_seq_len, vocab_size)
        ]:
        """Forward pass takes in x_t, t, padding mask, and condition, returns rates and probabilities
        """
        batch_size, x_seq_len = tokens.shape
        token_emb = self.token_embedding(tokens)    # (batch_size, x_seq_len, hidden_dim)

        time_emb = self.time_embedding(time_step)   # (batch_size, hidden_dim)
        time_emb = time_emb.unsqueeze(1).expand(-1, x_seq_len, -1)  # (batch_size, x_seq_len, hidden_dim)

        positions = torch.arange(x_seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)     # (batch_size, x_seq_len, hidden_dim)

        x = token_emb + time_emb + pos_emb          # (batch_size, x_seq_len, hidden_dim)
        x = x.transpose(0, 1)                       # expects (x_seq_len, batch_size, hidden_dim)

        # 处理条件：投影并转置
        cond_seq = self.cond_proj(condition)  # (batch, num_features, hidden_dim)
        cond_seq = cond_seq.transpose(0, 1)    # (num_features, batch, hidden_dim)

        # Transformer 层 + 交叉注意力
        for i, layer in enumerate(self.layers):
            x = layer(x, src_key_padding_mask=padding_mask)
            x = self.cross_attn_layers[i](x, cond_seq)

        x = x.transpose(0, 1)                           # (batch_size, x_seq_len, hidden_dim)
        x = self.final_layer_norm(x)                    # (batch_size, x_seq_len, hidden_dim)
        ins_logits = self.ins_logits_out(x)             # (batch_size, x_seq_len, vocab_size)
        sub_logits = self.sub_logits_out(x)             # (batch_size, x_seq_len, vocab_size)
        rates = F.softplus(self.rates_out(x))           # (batch_size, x_seq_len, 3) - ensure positive rates

        ins_probs = F.softmax(ins_logits, dim=-1)   # (batch_size, x_seq_len, vocab_size)
        sub_probs = F.softmax(sub_logits, dim=-1)   # (batch_size, x_seq_len, vocab_size)

        # Zero out outputs for padded positions
        mask_expanded = (~padding_mask).unsqueeze(-1).float()  # (batch_size, x_seq_len, 1)
        rates = rates * mask_expanded
        ins_probs = ins_probs * mask_expanded
        sub_probs = sub_probs * mask_expanded

        if torch.isnan(rates).any() or torch.isnan(ins_probs).any() or torch.isnan(sub_probs).any():
            raise ValueError("NaN detected in output probabilities or rates")

        return (
            rates,
            ins_probs,
            sub_probs,
        )
