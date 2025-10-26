# Main script to run training

import os
import torch
import pandas as pd

from config import *
from model import SecondaryStructurePredictor
from dataset import create_dataloader
from utils import (
    logger, linear_beta, setup_csv_logger, 
    log_metrics_to_csv
)
import time

def train_model(fam='5s', batch_size=None, accumulation_steps=None, use_amp=None, results_path=None):
    """Train the model for a specific RNA family"""
    # Load data splits
    df = pd.read_csv(f'data/ArchiveII.csv', index_col="id")
    splits = pd.read_csv(f'data/ArchiveII_famfold_splits.csv', index_col="id")
    
    # Filter data for the specific family
    train = df.loc[splits[(splits.fold==fam) & (splits.partition!="test")].index]
    test = df.loc[splits[(splits.fold==fam) & (splits.partition=="test")].index]
    
    # Use provided parameters or defaults from config
    if batch_size is None:
        batch_size = BATCH_SIZE
    if accumulation_steps is None:
        accumulation_steps = ACCUMULATION_STEPS
    if use_amp is None:
        use_amp = USE_AMP
    if results_path is None:
        results_path = RESULTS_PATH
    
    # # Create necessary directories
    data_path = f"data/{fam}"
    # os.makedirs(data_path, exist_ok=True)
    # os.makedirs(results_path, exist_ok=True)
    
    # # Save train and test data
    # train.to_csv(f"{data_path}/train.csv")
    # test.to_csv(f"{data_path}/test.csv")
    
    logger.info("+" * 80)
    logger.info(f"ArchiveII {fam} TRAINING STARTED".center(80))
    logger.info("+" * 80)

    # Initialize model
    net = SecondaryStructurePredictor(embed_dim=4, device=DEVICE, lr=LEARNING_RATE, use_amp=use_amp)
    
    # # Load pretrained weights if available
    # checkpoint_path = f"{RESULTS_PATH}/827weights.pmt"
    # net.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(DEVICE)))
    # logger.info(f"Loaded weights from {checkpoint_path}")
    
    # # Training settings
    # noise_added = True  # Flag to indicate if noise will be increased in the next epoch
    # first_noise_step_done = True  # Flag to indicate whether to add noise or not
    # previous_loss = 0.003921333109331389  # Loss reached by the saved model
    # best_loss_dict = [{"epoch": 827, "loss": 0.003921333109331389}]
    # t = INITIAL_NOISE_STEP  # Initial noise step

    # logger.info(f"Noise steps: {NOISE_STEPS}")
    # logger.info(f"Tolerance: {TOLERANCE}")
    logger.info(f"Max epochs: {MAX_EPOCHS}")
    # logger.info(f"Closeness percentage: {CLOSENESS_PERCENTAGE}")
    logger.info(f"Learning rate: {LEARNING_RATE}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Accumulation steps: {accumulation_steps}")
    logger.info(f"Effective batch size: {batch_size * accumulation_steps}")
    logger.info(f"Mixed Precision (AMP): {'Enabled' if use_amp else 'Disabled'}")
    logger.info(f"Device: {DEVICE}")
    # Setup CSV for logging metrics
    csv_path = os.path.join(results_path, "metrics.csv")
    fieldnames = [
        "train_loss", "train_f1", "train_contact_loss", "train_probing_loss", "train_f1_probing",
        # "val_loss", "val_f1", "val_contact_loss", "val_probing_loss", "val_f1_probing",
        "epoch_time_s"
        # "hard_test_loss", "hard_test_f1",
        # "noise_added", "beta",
        # "epoch",
        # "noise_step"
    ]
    # setup_csv_logger(csv_path, fieldnames)
    train_loader = create_dataloader(
        "one-hot",
        f"{data_path}/train.csv",
        "data/ArchiveII_probing.pt",
        batch_size,
        True,
    )

    # Validate on test set
    val_loader = create_dataloader(
        "one-hot",
        f"{data_path}/test.csv",
        "data/ArchiveII_probing.pt",
        batch_size,
        False,
    )

    # Training loop
    for epoch in range(1, MAX_EPOCHS):
        time_start = time.time()
        metrics = {}
        logger.info(f"Starting epoch {epoch}")

        # Train for one epoch with gradient accumulation
        metrics = net.fit(train_loader, accumulation_steps=accumulation_steps)
        metrics = {f"train_{k}": v for k, v in metrics.items()}
        
        logger.info("Running validation")
        val_metrics = net.test(val_loader)
        val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
        metrics.update(val_metrics)
        metrics.update({"epoch_time_s": time.time() - time_start})
        # Log metrics
        # log_metrics_to_csv(csv_path, metrics)
        logger.info(" ".join([f"{k}: {v}" for k, v in metrics.items()]))

    logger.info(f"ArchiveII {fam} TRAINING ENDED".center(80))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train RNA secondary structure prediction model')
    parser.add_argument('--family', type=str, default='5s', help='RNA family to train on')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size (default: from config)')
    parser.add_argument('--accumulation-steps', type=int, default=None, help='Gradient accumulation steps (default: from config)')
    parser.add_argument('--use-amp', action='store_true', default=None, help='Enable automatic mixed precision (FP16)')
    parser.add_argument('--no-amp', dest='use_amp', action='store_false', help='Disable automatic mixed precision')
    args = parser.parse_args()
    
    train_model(fam=args.family, batch_size=args.batch_size, accumulation_steps=args.accumulation_steps, use_amp=args.use_amp)
