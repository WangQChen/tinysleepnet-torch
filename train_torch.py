# train_torch.py (final version with logging and checkpointing)

import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from model_torch import TinySleepNet
from trainer_torch import Trainer
from logger_torch import setup_logger, save_config, backup_source_code
from sklearn.model_selection import train_test_split
import time

from dataset_torch import SleepEDFNPZ



def train_main(epoch=10):
    data_dir = './data/sleepedf/sleep-cassette/eeg_fpz_cz'
    all_files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
    train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)

    train_dataset = SleepEDFNPZ(train_files)
    test_dataset = SleepEDFNPZ(test_files)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

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

    # Setup logging
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_dir = f'logs/run_{timestamp}'
    logger = setup_logger(log_dir)
    save_config(config, log_dir)
    backup_source_code('./', log_dir)

    logger.info("Starting training...")

    model = TinySleepNet(config)
    trainer = Trainer(model, config, device='cuda' if torch.cuda.is_available() else 'cpu')

    for epoch in range(epoch):
        loss, acc, f1 = trainer.train_epoch(train_loader)
        logger.info(f"[Epoch {epoch+1}] Train Loss: {loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")

        val_loss, val_acc, val_f1 = trainer.evaluate(test_loader)
        logger.info(f"           Valid Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        ckpt_path = os.path.join(log_dir, f"checkpoint_epoch_{epoch+1}.pt")
        trainer.save_checkpoint(ckpt_path)


if __name__ == '__main__':
    train_main
