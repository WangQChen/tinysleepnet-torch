# predict.py (PyTorch version)

import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from model_torch import TinySleepNet, Trainer
from sklearn.metrics import classification_report, confusion_matrix


class SleepEDFNPZ(torch.utils.data.Dataset):
    def __init__(self, files):
        self.data, self.labels = [], []
        self.names = []
        for f in files:
            d = np.load(f)
            x = d['x'].squeeze(-1).squeeze(-1)  # (N, L)
            y = d['y']
            self.data.append(x)
            self.labels.append(y)
            self.names.extend([os.path.basename(f)] * len(y))
        self.data = torch.tensor(np.vstack(self.data), dtype=torch.float32).unsqueeze(1)  # (N, 1, L)
        self.labels = torch.tensor(np.hstack(self.labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.names[idx]


def predict(checkpoint_path, data_dir='./data/sleepedf/sleep-cassette/eeg_fpz_cz'):
    files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
    dataset = SleepEDFNPZ(files)
    dataloader = DataLoader(dataset, batch_size=64)

    config = {
        'input_size': 3000,
        'n_classes': 5,
        'sampling_rate': 100,
        'learning_rate': 1e-4,
        'use_rnn': False,
        'seq_length': 20,
        'n_rnn_units': 128,
        'n_rnn_layers': 1
    }

    model = TinySleepNet(config)
    trainer = Trainer(model, config, device='cuda' if torch.cuda.is_available() else 'cpu')
    trainer.load_checkpoint(checkpoint_path)

    model.eval()
    all_preds, all_trues, all_names = [], [], []
    with torch.no_grad():
        for x, y, name in dataloader:
            x = x.to(trainer.device)
            out = model(x)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_trues.extend(y.numpy())
            all_names.extend(name)

    print("Classification Report:")
    print(classification_report(all_trues, all_preds))

    print("Confusion Matrix:")
    print(confusion_matrix(all_trues, all_preds))

    np.savez("predictions.npz", y_true=all_trues, y_pred=all_preds, filenames=all_names)
    print("Predictions saved to predictions.npz")


if __name__ == '__main__':
    predict(checkpoint_path='./checkpoint_epoch_10.pt')
