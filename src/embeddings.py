# -*- coding: utf-8 -*-
"""토큰 임베딩 + 위치 임베딩 과제 템플릿."""

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """
    token ID를 Transformer 입력 벡터로 바꿉니다.

    구현할 구조:
    - token embedding: nn.Embedding(vocab_size, emb_dim)
    - position embedding: nn.Embedding(context_length, emb_dim)
    - token embedding + position embedding
    - dropout
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        context_length: int,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(context_length, emb_dim)
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor: # x: 문장의 단어 id값 배열 나옴
        """
        TODO: token embedding과 position embedding을 더한 뒤 dropout을 적용합니다.

        Args:
            x: (batch_size, seq_len) token IDs

        Returns:
            (batch_size, seq_len, emb_dim)
        """
        # token_embedding에서 한꺼번에 여러 문장 단어 뽑아오기: 
        # => 3차원 배열(배치 갯수, 스퀀스 길이, 차원 갯수)
        token_extracted = self.token_embedding(x)

        # position_embedding에서도 뽑아오기:
        # x의 문장 길이 => 증감 배열 => 뽑아오기
        seq_len = x.size(1)
        positions = torch.arange(seq_len)
        position_extracted = self.position_embedding(positions)

        # 합한 것을 반환하기
        return token_extracted + position_extracted