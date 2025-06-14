# logger.py (PyTorch version)

import os
import sys
import logging
import shutil
from datetime import datetime


def setup_logger(log_dir, log_name='train.log'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_name)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def save_config(config, log_dir):
    config_path = os.path.join(log_dir, 'config.txt')
    with open(config_path, 'w') as f:
        for key, val in config.items():
            f.write(f"{key}: {val}\n")


def backup_source_code(src_dir, log_dir):
    code_backup_dir = os.path.join(log_dir, 'code')
    if not os.path.exists(code_backup_dir):
        os.makedirs(code_backup_dir)
    for file_name in os.listdir(src_dir):
        if file_name.endswith('.py'):
            shutil.copy(os.path.join(src_dir, file_name), code_backup_dir)


# Example usage in train.py or trainer.py:
# logger = setup_logger('./logs')
# logger.info("Training started")
# save_config(config, './logs')
# backup_source_code('./', './logs')
