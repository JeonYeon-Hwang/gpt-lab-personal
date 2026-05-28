# -*- coding: utf-8 -*-
"""GPT 사전 학습용 Dataset/DataLoader 과제 템플릿."""

import torch
from torch.utils.data import DataLoader, Dataset


class GPTDataset(Dataset):
    """
    token ID 리스트를 다음 토큰 예측용 input/target 쌍으로 자릅니다.

    예: token_ids=[10, 11, 12, 13], context_length=3
    - input:  [10, 11, 12]
    - target: [11, 12, 13]
    """

    def __init__(
        self,
        token_ids: list[int],
        context_length: int,
        stride: int | None = None,
    ):
        self.token_ids = token_ids
        self.context_length = context_length
        self.stride = stride if stride is not None else context_length
        self._length = ((len(token_ids) - context_length - 1) // self.stride) + 1

    def __len__(self) -> int:
        """
        """
        return int(self._length)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            input_ids: (context_length,)
            target_ids: (context_length,)
        """
        pos = idx * self.stride
        input_ids = self.token_ids[pos : pos + self.context_length]
        target_ids = self.token_ids[pos + 1 : pos + self.context_length + 1]

        return (torch.tensor(input_ids, dtype=torch.long), 
                torch.tensor(target_ids, dtype=torch.long))

def create_dataloader(
    token_ids: list[int],
    context_length: int,
    batch_size: int = 8,
    stride: int | None = None,
    drop_last: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    """
    # dataset 객체를 만든다
    dataset = GPTDataset(token_ids, context_length, stride)

    # dataloader를 만든다
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

