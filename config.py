# Configuration parameters for the project

import torch

# Training parameters
BATCH_SIZE = 4
MAX_EPOCHS = 300
LEARNING_RATE = 1e-4

# Noise parameters
NOISE_STEPS = 100
TOLERANCE = 1e-5
CLOSENESS_PERCENTAGE = 0.01
INITIAL_NOISE_STEP = 41

# Paths
DATA_PATH = "data"
RESULTS_PATH = "results"

# Device configuration
DEVICE = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else 'cpu'
