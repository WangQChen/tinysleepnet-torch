import torch
import numpy as np

class SleepEDFNPZ(torch.utils.data.Dataset):
    def __init__(self, files):
        self.data, self.labels = [], []
        for f in files:
            d = np.load(f)
            x = d['x']
            if x.ndim == 3:  # (N, 1, L)
                x = x.squeeze(1)
            elif x.ndim == 2:  # (N, L)
                pass
            else:
                raise ValueError(f"Unexpected shape: {x.shape}")

            y = d['y']
            self.data.append(x)
            self.labels.append(y)

        self.data = torch.tensor(np.vstack(self.data), dtype=torch.float32).unsqueeze(1)  # (N, 1, L)
        self.labels = torch.tensor(np.hstack(self.labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]