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

def train_model(fam='5s'):
    """Train the model for a specific RNA family"""
    # Load data splits
    df = pd.read_csv(f'data/ArchiveII.csv', index_col="id")
    splits = pd.read_csv(f'data/ArchiveII_famfold_splits.csv', index_col="id")
    
    # Filter data for the specific family
    train = df.loc[splits[(splits.fold==fam) & (splits.partition!="test")].index]
    test = df.loc[splits[(splits.fold==fam) & (splits.partition=="test")].index]
    
    # Create necessary directories
    data_path = f"data/{fam}"
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    # Save train and test data
    train.to_csv(f"{data_path}/train.csv")
    test.to_csv(f"{data_path}/test.csv")
    
    logger.info("+" * 80)
    logger.info(f"ArchiveII {fam} TRAINING STARTED".center(80))
    logger.info("+" * 80)

    # Initialize model
    net = SecondaryStructurePredictor(embed_dim=4, device=DEVICE, lr=LEARNING_RATE)
    
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

    # Setup CSV for logging metrics
    csv_path = os.path.join(RESULTS_PATH, "metrics.csv")
    fieldnames = [
        "train_loss", "train_f1", "train_contact_loss", "train_probing_loss",
        "val_loss", "val_f1", "val_contact_loss", "val_probing_loss",
        # "hard_test_loss", "hard_test_f1",
        # "noise_added", "beta",
        # "epoch",
        # "noise_step"
    ]
    setup_csv_logger(csv_path, fieldnames)
    
    # Training loop
    for epoch in range(1, MAX_EPOCHS):
        metrics = {}
        logger.info(f"Starting epoch {epoch}")

        # # Calculate noise level (beta)
        # beta = linear_beta(0, t, 1, NOISE_STEPS)
        # if first_noise_step_done:
        #     if beta > 1:
        #         beta = 1
        # else:
        #     logger.info("Not adding noise")
        #     beta = 0
            
        # logger.info(f"Current noise step: {t:.2f}")
        # logger.info(f"Max noise steps: {NOISE_STEPS}")
        # logger.info(f"Beta: {beta:.6f}")

        # Create dataloaders with current noise level
        train_loader = create_dataloader(
            "one-hot",
            f"{data_path}/train.csv",
            "data/ArchiveII_probing.pt",
            BATCH_SIZE,
            True,
            # beta=beta,
        )
        
        # Train for one epoch
        metrics = net.fit(train_loader)
        metrics = {f"train_{k}": v for k, v in metrics.items()}

        # Validate on test set
        val_loader = create_dataloader(
            "one-hot",
            f"{data_path}/test.csv",
            "data/ArchiveII_probing.pt",
            int(len(test)/2),
            False,
            # beta=beta,
        )
        
        logger.info("Running validation")
        val_metrics = net.test(val_loader)
        val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
        metrics.update(val_metrics)

        # # Test with hard (maximum) noise
        # hard_test_loader = create_dataloader(
        #     "one-hot",
        #     f"{data_path}/test.csv",
        #     "data/ArchiveII_probing.pt",
        #     int(len(test)/2),
        #     False,
        #     beta=1,
        # )
        
        # logger.info("Running hard test")
        # hard_test_metrics = net.test(hard_test_loader)
        # hard_test_metrics = {f"hard_test_{k}": v for k, v in hard_test_metrics.items()}
        # metrics.update(hard_test_metrics)

        # # Add noise-related metrics
        # noise_metrics = {"noise_added": noise_added, "beta": beta, "epoch": epoch, "noise_step": t}
        # metrics.update(noise_metrics)

        # # Check if we need to add more noise
        # current_loss = metrics['train_loss']
        # best_loss = 0.003921333109331389
        # closeness_perc = (current_loss - best_loss) / best_loss
        # close_to_best = closeness_perc < CLOSENESS_PERCENTAGE
        # logger.info(f"Closeness percentage: {closeness_perc}")
        # logger.info(f"Close to best: {close_to_best}")

        # if current_loss - previous_loss > TOLERANCE:  # Loss worsened
        #     logger.info("Loss worsened, not adding noise")
        #     noise_added = False
        # elif first_noise_step_done and close_to_best:
        #     # Add noise since we're close to best performance
        #     logger.info(f"Passed warm up epochs and we are close to best, adding noise")
        #     noise_added = True
        #     first_noise_step_done = True
        #     t += 1

        #     # Save model checkpoint
        #     logger.info("Saving model")
        #     torch.save(
        #         net.state_dict(),
        #         f"{RESULTS_PATH}/{epoch}weights.pmt",
        #     )

        #     # if beta>0.39:
        #     #     logger.info("noise level above 0.39, lr is now 1e-3")
        #     #     lr=1e-3

        #     # Reset optimizer state
        #     logger.info("Resetting optimizer state")
        #     net.optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
            
        #     # logger.info("updating best loss")
        #     # best_loss_dict.append({"epoch": epoch, "loss": current_loss})
        #     logger.info(f"Last entry best loss dict: {best_loss_dict[-1]}")
        # else:
        #     logger.info("Loss improved, not adding noise")
        #     noise_added = False
        #     if not first_noise_step_done:  # Update best loss for the first epochs
        #         logger.info("Updating best loss")
        #         best_loss_dict.append({"epoch": epoch, "loss": current_loss})

        # previous_loss = current_loss

        # Log metrics
        log_metrics_to_csv(csv_path, metrics)
        logger.info(" ".join([f"{k}: {v}" for k, v in metrics.items()]))

    logger.info(f"ArchiveII {fam} TRAINING ENDED".center(80))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train RNA secondary structure prediction model')
    parser.add_argument('--family', type=str, default='5s', help='RNA family to train on')
    args = parser.parse_args()
    
    train_model(fam=args.family)
