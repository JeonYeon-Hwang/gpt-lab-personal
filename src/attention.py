# -*- coding: utf-8 -*-
"""Multi-Head Self-Attention 과제 템플릿."""

import torch
import torch.nn as nn

#추가 임포트
import math


class MultiHeadAttention(nn.Module):
    """
    GPT의 causal self-attention을 구현합니다.

    구현할 핵심:
    - Q/K/V projection
    - head 분리: (B, T, C) -> (B, n_heads, T, head_dim)
    - attention score = QK^T / sqrt(head_dim)
    - causal mask로 미래 토큰 가리기
    - attention weight와 V를 곱한 뒤 head를 다시 합치기
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # 원소 중 10%를 0으로 만드는 행렬 산출
        self.dropout = nn.Dropout(drop_rate)
        #qkv 벡터를 산출하는 행렬들 정의
        self.W_q = nn.Linear(d_model, d_model, qkv_bias)
        self.W_k = nn.Linear(d_model, d_model, qkv_bias)
        self.W_v = nn.Linear(d_model, d_model, qkv_bias)
        # attention 결과물 행렬 * output projection = head 융합된 행렬 산출
        self.out_proj = nn.Linear(d_model, d_model, bias=False)


    # 인자 해설:
    # x: 임베딩된 (단어별 고유값) 행렬
    # causal_mask: 뒷 단어 못보게
    # return_attention_weights: 추가(점수 행렬) 반환 여부
    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = True,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: multi-head attention forward를 구현합니다.

        Args:
            x: (batch_size, seq_len, d_model)
            causal_mask: True이면 미래 위치를 볼 수 없게 mask 처리
            return_attention_weights: True이면 attention weight도 함께 반환
        """
        # 재료용 변수 선언
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        heads = self.n_heads
        head_dim = x.shape[2] // heads
        scale = math.sqrt(head_dim)

        # 1. 각각 qkv 벡터를 산출한다: 3차원 형태(batch_size, seq_len, 128)
        vector_Q = self.W_q(x)
        vector_K = self.W_k(x)
        vector_V = self.W_v(x)

        # 2. 멀티 헤드 n개로 쪼갠다: 4차원 형태
        pre_multi_Q = vector_Q.view(batch_size, seq_len, heads, head_dim)
        pre_multi_K = vector_K.view(batch_size, seq_len, heads, head_dim)
        pre_multi_V = vector_V.view(batch_size, seq_len, heads, head_dim)

        # 3. 4차원의 순서를 바꾼다: 바꾼 순서 => (batch_size, heads, seq_len, head_dim)
        multi_Q = pre_multi_Q.transpose(1, 2)
        multi_K = pre_multi_K.transpose(1, 2)
        multi_V = pre_multi_V.transpose(1, 2)

        # 4. 내접 연산 진행: 행렬 곱셈을 위해 맞춘다
        #    Q(배치, 머리, 문장 길이, 머리 갯수) * K(배치, 머리, 머리 갯수, 문장 길이)
        pre_scores = torch.matmul(multi_Q, multi_K.transpose(-2, -1))

        # 5. 마스킹 하기 => softmax로 변환(128차원이 n개로).. (배치, 머리 갯수, 문장 길이, 문장 길이)
        if causal_mask:
            masks = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
            pre_scores = pre_scores.masked_fill(masks, float('-inf'))
        
        attention_weight = torch.softmax(pre_scores / scale, dim=-1) 

        # 6. 각 단어(X) 별 델타E 를 구한다 
        #    순서 변경: (배치, 문장 길이, 머리 갯수, head_dim)
        #    차원 낮춤: 4 => 3차원, 뒤의 두 행렬을 곱함
        delta_E = torch.matmul(attention_weight, multi_V)
        delta_E = delta_E.transpose(1, 2).contiguous()
        delta_E = delta_E.view(delta_E.size(0), delta_E.size(1), -1)

        # 7. 드롭아웃 등 진행
        out_E = self.out_proj(delta_E)
        out_E = self.dropout(out_E)

        if return_attention_weights:
            return out_E, attention_weight
        else:
            return out_E
