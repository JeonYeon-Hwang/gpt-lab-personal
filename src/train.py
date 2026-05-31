# -*- coding: utf-8 -*-
"""GPT 사전 학습 유틸리티 과제 템플릿."""

import matplotlib.pyplot as plt
import torch

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel

# 인자값 설명:
# 현 문장 데이터(배치 갯수, 문장 길이), 정답 문장(한 단어가 더 있음), 모델 본체, 하드웨어 객체  
def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: GPTModel,
    device: torch.device,
) -> torch.Tensor:
    """"""
    # 데이터를 특정 하드웨어에서 연산하도록 지정
    inputs = input_batch.to(device)
    targets = target_batch.to(device)

    # 모델 실행
    logits = model(inputs)

    # loss(cross_entropy) 산출하기: 2차원(logits)과 1차원(targets) 비교
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), targets.flatten()
    )

    return loss


# 인자값 설명: 
# data_loader => 각 배치를 감싼 객체[(inputs, targets), ..], num_batches => 최대 몇 개?  
def calc_loss_loader(
    data_loader,
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """TODO: data_loader의 평균 loss를 계산합니다. 검증에서는 torch.no_grad()를 사용하세요."""
    total_loss = 0
    batch_cnt = 0

    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            # loss 산출
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
            batch_cnt += 1

            # 탈출 조건
            if num_batches and batch_cnt >= num_batches:
                break

    # 평균치 반환
    return total_loss / batch_cnt

def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
) -> None:
    """TODO: model/optimizer 상태, epoch, global_step을 torch.save로 저장합니다."""
    raise NotImplementedError("save_checkpoint를 구현하세요.")


def load_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device,
) -> tuple[int, int]:
    """TODO: torch.load로 checkpoint를 읽어 model/optimizer 상태를 복원합니다."""
    raise NotImplementedError("load_checkpoint를 구현하세요.")


def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """TODO: temperature와 top-k 샘플링을 지원하는 생성 함수를 구현합니다."""
    raise NotImplementedError("generate를 구현하세요.")


def generate_and_print_sample(
    model: GPTModel,
    tokenizer,
    device: torch.device,
    start_context: str,
    max_new_tokens: int = 50,
    context_size: int = 256,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> None:
    """TODO: start_context를 encode하고 generate 후 decode하여 출력합니다."""
    raise NotImplementedError("generate_and_print_sample을 구현하세요.")


def train_model(
    model: GPTModel,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer,
    ckpt_freq: int | None = None,
    start_epoch: int = 0,
    global_step: int = 0,
) -> list[float]:
    """TODO: 사전 학습 루프를 구현하고 epoch별 train loss 리스트를 반환합니다."""
    raise NotImplementedError("train_model을 구현하세요.")


def plot_losses(train_losses: list[float], val_losses: list[float] | None = None) -> None:
    """훈련/검증 손실 그래프를 그리는 제공 함수."""
    plt.plot(train_losses, label="Train")
    if val_losses is not None:
        plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training / Validation Loss")
    plt.show()
