# trainer.py (PyTorch version)

import os
import torch
from sklearn import metrics


class Trainer:
    def __init__(self, model, config, device='cpu'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            B, T = y.shape if y.ndim == 2 else (None, None)

            self.optimizer.zero_grad()
            out = self.model(x)
            if B:  # RNN mode
                out = out.view(B * T, -1)
                y = y.view(-1)

            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            all_preds.extend(out.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        acc = metrics.accuracy_score(all_labels, all_preds)
        f1 = metrics.f1_score(all_labels, all_preds, average='macro')
        return total_loss / len(dataloader), acc, f1

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                B, T = y.shape if y.ndim == 2 else (None, None)
                out = self.model(x)
                if B:
                    out = out.view(B * T, -1)
                    y = y.view(-1)

                loss = self.criterion(out, y)
                total_loss += loss.item()
                all_preds.extend(out.argmax(dim=1).cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        acc = metrics.accuracy_score(all_labels, all_preds)
        f1 = metrics.f1_score(all_labels, all_preds, average='macro')
        return total_loss / len(dataloader), acc, f1

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
