# -*- coding: utf-8 -*-
"""GPT 모델 구성 요소 과제 템플릿."""

import torch
import torch.nn as nn

try:
    from .attention import MultiHeadAttention
    from .embeddings import InputEmbedding
except ImportError:
    from attention import MultiHeadAttention
    from embeddings import InputEmbedding


class LayerNorm(nn.Module):
    """마지막 차원 기준 Layer Normalization."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        # x: 3차원(batch_size, Sequence_len, emb_dim)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        return self.gamma * norm_x + self.beta


class GELU(nn.Module):
    """GPT FeedForward에서 사용하는 GELU 활성화 함수."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(0.7978845608028654 *
                                        (x + 0.044715 * torch.pow(x, 3))))


class QuickGELU(nn.Module):
    """quick_gelu: x * sigmoid(1.702 * x)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


def build_activation(activation_name: str = "gelu") -> nn.Module:
    """FeedForward에서 사용할 activation module을 생성합니다."""
    if activation_name == "gelu":
        return GELU()
    if activation_name == "gelu_exact":
        return nn.GELU(approximate="none")
    if activation_name == "quick_gelu":
        return QuickGELU()

    raise ValueError(
        "activation_name must be one of: gelu, gelu_exact, quick_gelu"
    )


class FeedForward(nn.Module):
    """Transformer FFN: Linear -> GELU -> Linear -> Dropout."""

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        mult: int = 4,
        activation_name: str = "gelu",
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_model * mult),
            build_activation(activation_name),
            nn.Linear(d_model * mult, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    GPT block: LayerNorm -> Causal Self-Attention -> residual,
    LayerNorm -> FeedForward -> residual.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
        activation_name: str = "gelu",
    ):
        super().__init__()
        self.att = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            drop_rate=drop_rate,
            qkv_bias=qkv_bias)
        
        self.ff = FeedForward(
            d_model=d_model,
            dropout=drop_rate,
            activation_name=activation_name)
        
        self.norm1 = LayerNorm(normalized_shape=d_model)
        self.norm2 = LayerNorm(normalized_shape=d_model)
        
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        """"""
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    """InputEmbedding -> TransformerBlock N개 -> LayerNorm -> LM head."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        v_size = config["vocab_size"]
        e_dim = config["emb_dim"]
        context_len = config["context_length"]
        drop_rate = config["drop_rate"]
        n_layers = config["n_layers"]
        n_heads = config["n_heads"]
        qkv_bias = config["qkv_bias"]
        activation_name = config.get("activation_name", "gelu")

        self.tok_emb = nn.Embedding(v_size, e_dim)
        self.pos_emb = nn.Embedding(context_len, e_dim)
        self.drop_emb = nn.Dropout(drop_rate)

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(e_dim, n_heads, drop_rate, qkv_bias, activation_name) 
              for _ in range(n_layers)]
        )

        self.final_norm = LayerNorm(e_dim)
        self.out_head = nn.Linear(e_dim, v_size, bias=False)

        self.loss_fn = nn.CrossEntropyLoss()


    # idx: 현재 입력된 단어들의 ID를 의미
    # targets: 정답지(training용) => 없으면: 실제 구동 상태, 있으면: loss 반환(역전파)
    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            targets가 None이면 logits
            targets가 있으면 (loss, logits)
        """
        # 1. idx의 행(문장 갯수)렬(문장 길이)를 구한다
        batch_size, seq_len = idx.shape
        
        # 2. idx 기반으로 임베딩 행렬을 생성한다
        #    또한, 위치 행렬도 생성한다  
        #    이 둘을 합해 x 생성 
        tok_embeds = self.tok_emb(idx)
        pos_indices = torch.arange(seq_len, device=idx.device)
        pos_embeds = self.pos_emb(pos_indices)
        x = tok_embeds + pos_embeds

        # 3. dropout 진행
        # 4. 트랜스포머 블록 통과: N번
        # 5. 마지막 정규화 진행
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        # 6. logits 산출: 압축 차원 
        #    => 총 단어 차원만큼 확장(Batch_Size, Context_Length, Vocab_Size)
        logits = self.out_head(x)

        if targets is not None:
            # 평탄화(차원 줄이기) 진행  
            flat_logits = logits.view(-1, logits.size(-1))
            flat_targets = targets.view(-1)

            # loss 산출
            loss = self.loss_fn(flat_logits, flat_targets)

            return loss, logits
        else:
            return logits

# 인자값 설명:
# 모델 본체, 문장(배치, 현 문장 길이), 붙일 단어 최대 수, 최대 문맥 길이 
def generate_text_simple(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """"""
    
    for _ in range(max_new_tokens):
        idx_cnt = idx[:, -context_size:]

        # 과정에서 gradient 산출하지 말고 바로 연산할 것: 역전파를 하지 않겠다 뜻  
        with torch.no_grad():
            logits = model(idx_cnt)

        # 마지막 단어 점수판만 추출: (B, C_T_L, V_C) => (B_S, V_S)
        # 마지막 행을 기준으로 확률값으로 산출  
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)

        # 마지막 차원(dim-1) 기준으로 max인 점수인 ID를 가져오고, 2차원 형태는 유지
        # 기존 id 행렬 뒤에 붙이기: 여러 문장의 뒷 단어를 한꺼번에
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
