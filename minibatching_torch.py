# minibatching.py (PyTorch version - sequence batching for RNN mode)

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """
    Converts raw epochs into sequences for RNN training.
    Input shape: (N, 1, L), where N is #epochs.
    Output: (B, seq_len, 1, L), (B, seq_len) label
    """
    def __init__(self, npz_paths, seq_len=20):
        self.x, self.y = [], []
        for path in npz_paths:
            d = np.load(path)
            x = d['x'].squeeze(-1).squeeze(-1)  # (N, L)
            y = d['y']
            x = x.astype(np.float32)
            y = y.astype(np.int64)

            n_seq = len(y) // seq_len
            if n_seq == 0:
                continue

            x = x[:n_seq * seq_len].reshape(n_seq, seq_len, -1)
            y = y[:n_seq * seq_len].reshape(n_seq, seq_len)

            self.x.append(x)
            self.y.append(y)

        self.x = torch.tensor(np.concatenate(self.x), dtype=torch.float32).unsqueeze(2)  # (B, T, 1, L)
        self.y = torch.tensor(np.concatenate(self.y), dtype=torch.long)  # (B, T)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# 用法示例（在 train.py 中）
# from minibatching import SequenceDataset
# dataset = SequenceDataset(file_list, seq_len=20)
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True)  # 注意 batch_size 是 sequence 数
