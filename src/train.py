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
    """"""
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


# 인자값 설명:
# optimizer: 최적화 상태.. 학습 관성 정보, global_step: 총 진행한 batch 갯수, path: 저장 경로
def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
) -> None:
    """"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step
    }, path)


def load_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device,
) -> tuple[int, int]:
    """"""
    # 1. 체크포인트 불러오기
    # 2. 모델 가중치 복원
    # 3. optimizer 관성 복원 
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # 4. 나머지 정보는 추출 => 반환
    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]

    return epoch, global_step


# logits기반 실제 단어 생성 로직: 말 그대로 생성만
# 인자값 해설: idx(토큰 2차원 배열: 문장 n개), max~: 최대 붙일 갯수, eos_id: 이 번호 뽑힐 시 중단 
def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """"""
    # max가 될 때 까지 토큰 생성
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        # 모델 구동
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # top_k 로 logit K개로 필터
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            # min 보다 점수가 낮은 애들은 -무한으로 만들어 확률 0% 처리
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        
        # 온도에 따른 무작위성 증가 => 확률 변환 => 한 개 뽑기
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # 탈출조건 추가
        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


# 인자값 해설:
# tokenizer: 토큰 인코딩/디코딩, start_context: 시작 하는 문장들
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
    """"""
    # 1. encoding 실행 => 특정 하드웨어에 집어넣기
    encoded_idx = tokenizer.encode(start_context)
    encoded_idx.to(device)

    # 2. generate 함수 가동(연산).. 특정 하드웨어에서 수행
    with torch.no_grad():
        idx = generate(model, 
                    encoded_idx, 
                    max_new_tokens, 
                    context_size, 
                    temperature, 
                    top_k, 
                    tokenizer.encode("<eos>")[0])
    
    # 3. idx를 decode하고 print 실행
    decoded_text = tokenizer.decode(idx[0])
    print(decoded_text)


# 인자값 해설:
# train_loader: 전체 글을 특정 size로 쪼개서 줌
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
    """"""
    # epoch 수 만큼 반복
    for epoch in range(start_epoch, num_epochs):

        # 전체 글을 batch_data 만큼 쪼개서 진행
        for input_batch, target_batch in train_loader:
            # 1. loss 점수 산출
            loss = calc_loss_batch(input_batch, target_batch, model, device)

            # 2. 이전 gradient 청소 => 새 loss로 미분 진행 => 가중치 수정
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 4. 특정 주기 마다 실행
            global_step += 1
            if global_step % eval_freq == 0:

                # train_loss: 안쪽 for문 기준 방금 사용한 데이터 기준 산정
                # val_loss: 새롭게 학습한 단어 기준 산정
                train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
                val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_freq)
                
                print(f"Step {global_step}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}")
                generate_and_print_sample(model, tokenizer, device, start_context)

            # 5. 특정 주기 마다 체크포인트(save point) 저장
            if ckpt_freq is not None and global_step % ckpt_freq == 0:
                # torch에 저장 실행
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                }, f"checkpoint_step_{global_step}.pth")
                print(f"--- Step {global_step}: checkpoint save completed")


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
