#!/usr/bin/env python3
"""
Script to run training experiments with different batch sizes.
Handles out-of-memory exceptions and creates separate metrics files for each batch size.
"""

import os
import sys
import torch
import argparse
from main import train_model
from config import RESULTS_PATH
from utils import logger

def run_experiments(batch_sizes, accumulation_steps_list=None, use_amp=False, family='5s'):
    """
    Run training experiments with different batch sizes and accumulation steps.
    
    Args:
        batch_sizes: List of batch sizes to experiment with
        accumulation_steps_list: List of accumulation steps (must match length of batch_sizes)
                                If None, defaults to 1 for each batch size
        use_amp: Whether to use automatic mixed precision (FP16)
        family: RNA family to train on (default: '5s')
    """
    
    # If no accumulation steps provided, default to 1 for each batch size
    if accumulation_steps_list is None:
        accumulation_steps_list = [1] * len(batch_sizes)
    
    # Validate that lists have the same length
    if len(batch_sizes) != len(accumulation_steps_list):
        logger.error(f"batch_sizes and accumulation_steps must have the same length. "
                    f"Got {len(batch_sizes)} batch sizes and {len(accumulation_steps_list)} accumulation steps")
        sys.exit(1)
    
    logger.info("=" * 80)
    logger.info(f"STARTING BATCH SIZE EXPERIMENTS FOR FAMILY {family}".center(80))
    logger.info("=" * 80)
    logger.info(f"Batch sizes to test: {batch_sizes}")
    logger.info(f"Accumulation steps to test: {accumulation_steps_list}")
    logger.info(f"Effective batch sizes: {[bs * acc for bs, acc in zip(batch_sizes, accumulation_steps_list)]}")
    logger.info(f"Mixed Precision (AMP): {'Enabled' if use_amp else 'Disabled'}")
    
    successful_experiments = []
    failed_experiments = []
    
    for batch_size, accumulation_steps in zip(batch_sizes, accumulation_steps_list):
        try:
            effective_batch_size = batch_size * accumulation_steps
            logger.info("=" * 60)
            logger.info(f"EXPERIMENT: BATCH SIZE {batch_size} x ACCUM {accumulation_steps} = {effective_batch_size}".center(60))
            logger.info("=" * 60)
            
            # Set results path for this experiment
            if accumulation_steps > 1:
                experiment_results_path = f"{RESULTS_PATH}/batch_size_{batch_size}_accum_{accumulation_steps}"
            else:
                experiment_results_path = f"{RESULTS_PATH}/batch_size_{batch_size}"
            if use_amp:
                experiment_results_path += "_amp"
            
            # Create results directory for this batch size
            os.makedirs(experiment_results_path, exist_ok=True)
            
            logger.info(f"Running experiment with batch size: {batch_size}")
            logger.info(f"Accumulation steps: {accumulation_steps}")
            logger.info(f"Effective batch size: {effective_batch_size}")
            logger.info(f"Results will be saved to: {experiment_results_path}")
            
            # Run the training with specific batch size, accumulation steps, AMP, and results path
            train_model(fam=family, batch_size=batch_size, accumulation_steps=accumulation_steps, 
                       use_amp=use_amp, results_path=experiment_results_path)
            
            successful_experiments.append((batch_size, accumulation_steps, effective_batch_size))
            logger.info(f"✓ Successfully completed experiment with batch size {batch_size} x {accumulation_steps}")
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"✗ Out of memory error with batch size {batch_size} x {accumulation_steps}: {e}")
            failed_experiments.append((batch_size, accumulation_steps, "OutOfMemoryError"))
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Breaking all experiments due to memory constraints.")
            break
            
        except Exception as e:
            logger.error(f"✗ Unexpected error with batch size {batch_size} x {accumulation_steps}: {e}")
            failed_experiments.append((batch_size, accumulation_steps, str(e)))
            
            # Continue with next batch size for other errors
            continue
    
    # Summary
    logger.info("=" * 80)
    logger.info("EXPERIMENT SUMMARY".center(80))
    logger.info("=" * 80)
    
    if successful_experiments:
        logger.info(f"✓ Successful experiments:")
        for batch_size, accumulation_steps, effective_batch_size in successful_experiments:
            if accumulation_steps > 1:
                metrics_path = f"{RESULTS_PATH}/batch_size_{batch_size}_accum_{accumulation_steps}/metrics.csv"
            else:
                metrics_path = f"{RESULTS_PATH}/batch_size_{batch_size}/metrics.csv"
            logger.info(f"  - Batch size {batch_size} x {accumulation_steps} (effective: {effective_batch_size}): {metrics_path}")
    
    if failed_experiments:
        logger.info(f"✗ Failed experiments:")
        for batch_size, accumulation_steps, error in failed_experiments:
            logger.info(f"  - Batch size {batch_size} x {accumulation_steps}: {error}")
    
    logger.info("=" * 80)
    logger.info("ALL EXPERIMENTS COMPLETED".center(80))
    logger.info("=" * 80)

def main():
    parser = argparse.ArgumentParser(
        description='Run training experiments with different batch sizes and accumulation steps'
    )
    parser.add_argument(
        '--batch_sizes', 
        nargs='+', 
        type=int, 
        default=[4, 8, 16],
        help='List of batch sizes to experiment with (default: [4, 8, 16])'
    )
    parser.add_argument(
        '--accumulation_steps', 
        nargs='+', 
        type=int, 
        default=None,
        help='List of accumulation steps (must match length of batch_sizes). If not provided, defaults to 1 for each batch size'
    )
    parser.add_argument(
        '--use-amp',
        action='store_true',
        default=False,
        help='Enable automatic mixed precision (FP16) for all experiments'
    )
    parser.add_argument(
        '--family', 
        type=str, 
        default='5s',
        help='RNA family to train on (default: 5s)'
    )
    
    args = parser.parse_args()
    
    # Validate batch sizes
    if not all(bs > 0 for bs in args.batch_sizes):
        logger.error("All batch sizes must be positive integers")
        sys.exit(1)
    
    # Validate accumulation steps if provided
    if args.accumulation_steps is not None:
        if not all(acc > 0 for acc in args.accumulation_steps):
            logger.error("All accumulation steps must be positive integers")
            sys.exit(1)
        if len(args.batch_sizes) != len(args.accumulation_steps):
            logger.error(f"Number of batch_sizes ({len(args.batch_sizes)}) must match "
                        f"number of accumulation_steps ({len(args.accumulation_steps)})")
            sys.exit(1)
    
    batch_sizes = args.batch_sizes
    accumulation_steps_list = args.accumulation_steps
    
    run_experiments(batch_sizes, accumulation_steps_list, args.use_amp, args.family)

if __name__ == "__main__":
    main()
