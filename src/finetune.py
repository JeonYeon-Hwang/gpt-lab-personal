# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 과제 템플릿."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import csv
import random

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


# 인자값 설명:
# train용 원본 경로, test용 원본 경로, train중 몇 프로를 val로 쓸 지 결정, 랜덤 번호?
def make_sentiment_dataset(
    train_tsv_path: str | Path,
    test_tsv_path: str | Path | None = None,
    val_ratio: float = 0.08,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    반환 형식:
        [{"text": "리뷰", "label": 0 또는 1}, ...]
    """
    # path를 기준으로 원본 데이터를 불러온다
    temp_data = refine_raw_data(train_tsv_path)
    test_data = refine_raw_data(test_tsv_path)

    # val_ratio 기반으로 val 뽑아낸다
    random.seed(seed)
    random.shuffle(temp_data)
    val_len = int(len(temp_data) * val_ratio)

    # 이를 기준으로 train/val 나누기
    train_data = temp_data[:-val_len]
    val_data = temp_data[-val_len:]

    # output_dir가 있을 경우, 해당 경로에 저장하기: 현재 미구현
    if output_dir is not None:
        pass

    return train_data, val_data, test_data
    

class ReviewSentimentDataset(Dataset):
    """감성 분류용 Dataset. 리뷰 하나와 label 하나를 반환합니다."""

    def __init__(
        self,
        data: list[dict],
        tokenizer,
        max_length: int = 128,
        pad_id: int | None = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.get_pad_id() if pad_id is None else pad_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """"""
        # data의 특정 idx의 text와 label 뽑기
        query = self.data[idx]
        text = query["text"]
        label = query["label"]

        # 토큰화 하기 => max_length로 자르기 => 나머지 padding 하기 => 텐서화 
        ids = self.tokenizer.encode(text)
        ids = ids[:self.max_length]
        ids = ids + [self.pad_id] * (self.max_length - len(ids))
        input_ids = torch.tensor(ids, dtype=torch.long)

        # 라벨도 텐서화 하기
        label = torch.tensor(label, dtype=torch.long)

        return input_ids, label


# 문장 분류용 gpt로의 튜닝을 함 => classifier를 붙임
# 인자 설명: dropout: 과적합 방지
class GPTForSequenceClassification(nn.Module):
    """
    GPT backbone 위에 감성 분류용 Linear head를 붙인 모델.

    주의: LM head는 다음 토큰 예측용입니다. 감성 분류는 hidden state 위에 별도 classifier를 붙입니다.
    """

    def __init__(
        self,
        gpt_model: GPTModel,
        num_labels: int = 2,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.gpt = gpt_model
        self.num_labels = num_labels
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(gpt_model.config["emb_dim"], num_labels)

    # 
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.
        """
        batch_size, seq_len = input_ids.shape

        # input_ids에서 위치/임베딩 데이터 뽑아오기
        tok_embeds = self.gpt.tok_emb(input_ids)
        pos_indices = torch.arange(seq_len, device=input_ids.device)
        pos_embeds = self.gpt.pos_emb(pos_indices)

        # gpt 모델의 LM_head 이전 단계까지 불러오기
        x = tok_embeds + pos_embeds
        x = self.gpt.drop_emb(x)
        x = self.gpt.trf_blocks(x)
        hidden_states = self.gpt.final_norm(x)

        # 두 번째 요소 기준 마지막 가져오기: (batch, 단어 수, 차원 수) => (batch, 차원 수)
        last_token_vector = hidden_states[:, -1, :]
        # 1. dropout: 무작위로 차원 중 10%를 0으로 지움
        # 2. classifier: 차원을 2개로 강제로 줄임
        logits = self.classifier(self.dropout(last_token_vector))

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

            return loss, logits
        else:
            return logits
        


def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """"""
    # 변수 선언
    total_loss = 0
    correct = 0
    total = 0

    # 한 개의 epoch 기준 시행
    for input_batch, target_batch in train_loader:
        # loss 계산 진행
        inputs = input_batch.to(device)
        targets = target_batch.to(device)

        logits = model(inputs)

        loss = torch.nn.functional.cross_entropy(logits, targets)

        # 가산하기
        total_loss += loss.item()
        
        # 1. 가장 큰 차원 값 뽑기
        preds = logits.argmax(dim=-1)
        # 2. 맞힌 갯수 산출하기: True = 1, False = 0
        correct += (preds == targets).sum().item()
        # 3. 문재 갯수 가산하기
        total += targets.numel()

        # 청소 => 미분 => 수정
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

    # 최종 산출
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total

    return avg_loss, accuracy


# train 하지 않고 평가 모드로 바꿈
def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
) -> tuple[float, float]:
    """"""
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    # no_grad() => gradient 산출 비활성화: 이후 로직은 동일
    with torch.no_grad():
        for input_batch, target_batch in data_loader:
            inputs = input_batch.to(device)
            targets = target_batch.to(device)

            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits, targets)

            total_loss += loss.item()

            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.numel()

    # 최종 산출 로직은 동일하다
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total

    return avg_loss, accuracy


# 헬퍼 함수입니다
def refine_raw_data(tsv_path: str):
    refined_data = []

    with open(tsv_path, "r", encoding="utf-8") as f:
        # 읽기 객체 생성
        reader = csv.reader(f, delimiter="\t")
        next(reader)

        # 줄 단위로 읽기
        for row in reader:
            if len(row) < 3:
                continue

            text = row[1].strip()
            if not text:
                continue

            refined_data.append({"text": row[1], "label": int(row[2])})

    return refined_data
    
