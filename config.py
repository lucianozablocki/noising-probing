# Configuration parameters for the project

import torch

# Training parameters
BATCH_SIZE = 4  # Reduced from 16 to save memory
ACCUMULATION_STEPS = 4  # Effective batch size = 4 * 4 = 16
MAX_EPOCHS = 10
LEARNING_RATE = 1e-3
USE_AMP = False  # Automatic Mixed Precision (FP16) - set to True to enable

# # Noise parameters
# NOISE_STEPS = 100
# TOLERANCE = 1e-5
# CLOSENESS_PERCENTAGE = 0.1
# INITIAL_NOISE_STEP = 41

# Paths
DATA_PATH = "data"
RESULTS_PATH = "results"

# Device configuration
DEVICE = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else 'cpu'
