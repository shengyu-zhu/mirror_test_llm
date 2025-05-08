#!/usr/bin/env python3
"""
Step 3: Recognition test - Ask model to identify the marked sentence.
Handles API processing and error management.
Supports processing multiple prompt variants for selected model pairs.
"""
import os
import sys
import random
import pandas as pd
import asyncio
import time
import argparse
import json
import signal
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.util import *

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process mirror test results.')
parser.add_argument('--resume', action='store_true',
                    help='Resume processing from the last checkpoint')
parser.add_argument('--variant', type=str, default='alternative_m3_full_sentence',
                    choices=['alternative_m3_full_sentence', 'alternative_m3_cot', 
                             'alternative_m3_allow_0', 'alternative_m3_m1_unchanged',
                             'alternative_m3_numbered_sentences', 'alternative_m3_revealed_recognition_task'],
                    help='Select prompt variant to process')
parser.add_argument('--process-all-m3-variants', action='store_true',
                    help='Process all prompt variants for Grok-Gemini pair')
args = parser.parse_args()

# Set constants and configuration
RANDOM_SEED = 2025
INPUT_DATA_PATH = "./data/step3/input/"
OUTPUT_DATA_PATH = "./data/step3/output/"
CHECKPOINT_DIR = "./data/step3/checkpoints/"
CHECKPOINT_INTERVAL = 10

# Set MAX_CONCURRENT_TASKS based on arguments
if args.process_all_m3_variants:
    MAX_CONCURRENT_TASKS = 2
else:
    MAX_CONCURRENT_TASKS = 6

# Define variant-specific paths
VARIANT_OUTPUT_PATHS = {
    "alternative_m3_full_sentence": "./data/step3/output_full_sentence/",
    "alternative_m3_cot": "./data/step3/output_cot/",
    "alternative_m3_allow_0": "./data/step3/output_allow_0/",
    "alternative_m3_m1_unchanged": "./data/step3/output_m1_unchanged/",
    "alternative_m3_numbered_sentences": "./data/step3/output_numbered_sentences/",
    "alternative_m3_revealed_recognition_task": "./data/step3/output_revealed_recognition_task/"
}

# Make sure all output paths exist
for path in VARIANT_OUTPUT_PATHS.values():
    os.makedirs(path, exist_ok=True)

# Set the appropriate input column and output path based on arguments
if args.process_all_m3_variants:
    print("Processing all prompt variants for Grok-Gemini pair")
    # When processing all variants, we'll set these dynamically per variant
    INPUT_COLUMN = None  
    ACTIVE_OUTPUT_PATH = None
else:
    # Use the specified variant
    INPUT_COLUMN = f"step3_input_{args.variant}"  # Added "step3_input_" prefix to match column name format
    ACTIVE_OUTPUT_PATH = VARIANT_OUTPUT_PATHS[args.variant]
    print(f"Using variant input column: '{INPUT_COLUMN}'")
    print(f"Output will be saved to: {ACTIVE_OUTPUT_PATH}")

# Set random seed for reproducibility
random.seed(RANDOM_SEED)

# Define models registry for easy access
MODEL_REGISTRY = {
    "ChatGPT": ChatGPT,
    "Claude": Claude,
    "Grok": Grok,
    "Gemini": Gemini,
    "Llama": Llama,
    "Deepseek": Deepseek
}

# Use all models (removed --models argument)
models = list(MODEL_REGISTRY.values())

# Text processor for sentence operations
text_processor = TextProcessor()

# Global parameters
time_sleep_buffer = 0.1  # Buffer time between API calls
m1_temperature = MODEL_TEMPERATURES.get("default", 0.7)  # Set temperature for model responses

# Create necessary directories
os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Add signal handler for graceful shutdown
def signal_handler(sig, frame):
    print('\nReceived interrupt signal. Completing current batch before exiting...')
    # Set a flag to tell asyncio to stop scheduling new tasks
    signal_handler.shutdown_requested = True

# Initialize the flag
signal_handler.shutdown_requested = False

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

async def process_model_pair(m1, m2, variant='alternative_m3_full_sentence'):
    """
    Process a single model pair: have model m1 try to identify which sentence
    was modified in stories originally created by model m1 and marked by model m2.
    
    Args:
        m1: First model function (recognition model)
        m2: Second model function (story source model)
        variant: The prompt variant to use
        
    Returns:
        bool: True if processing succeeded, False otherwise
    """
    model1 = m1.__name__
    model2 = m2.__name__
    
    # Skip non-Grok-Gemini pairs when processing variants
    if args.process_all_m3_variants and (model1 != "Grok" or model2 != "Gemini"):
        return True
    
    # Set active output path and input column for this variant
    active_output_path = VARIANT_OUTPUT_PATHS[variant]
    input_column = f"step3_input_{variant}"  # Changed to match the actual column name format
    
    print(f"Processing Step 3: {model1} vs {model2} - Variant: {variant}")
    
    # Define input and output file paths
    input_file = f"{INPUT_DATA_PATH}mirror_test_results_{model1}_{model2}_processed.csv"
    output_file = f"{active_output_path}mirror_test_results_{model1}_{model2}_{variant}.csv"
    checkpoint_file = f"{CHECKPOINT_DIR}mirror_test_results_{variant}_{model1}_{model2}_{variant}_checkpoint.csv"
    
    # Skip if output file already exists and not resuming
    if os.path.exists(output_file) and not args.resume:
        print(f"File {output_file} already exists. Skipping this pair-variant.")
        return True
    
    # Skip if input file doesn't exist
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist. Skipping this pair-variant.")
        return False
    
    try:
        # Load the processed data from step 2
        df = pd.read_csv(input_file, sep='|')
        print(f"Loaded {len(df)} records from {input_file}")
        
        # Check if the specified input column exists
        if input_column not in df.columns:
            available_cols = [col for col in df.columns if col.startswith('step3_input_')]
            print(f"Error: Column '{input_column}' not found in the input file.")
            print(f"Available step3 input columns: {available_cols}")
            
            # Try to suggest alternative column names
            if len(available_cols) > 0:
                print(f"Did you mean one of these columns? {available_cols}")
                print(f"Attempting to use '{available_cols[0]}' instead...")
                input_column = available_cols[0]
            else:
                print(f"No suitable input columns found. Skipping this pair-variant.")
                return False
        
        # Define column names based on variant
        output_column = f"step3_output_{variant}"
        message_column = f"step3_output_{variant}_message_only"
        
        # Check for existing checkpoint or output
        completed_indices = []
        
        if args.resume:
            # Load checkpoint and output file data
            completed_indices = load_checkpoints(df, checkpoint_file, output_file, output_column, message_column)
        
        # Make sure we have these columns, creating them if needed
        if output_column not in df.columns:
            df[output_column] = None
        if message_column not in df.columns:
            df[message_column] = None
            
        # Clean completed indices - make sure they're all valid
        completed_indices = [idx for idx in completed_indices if idx < len(df)]
        
        # Remaining indices to process
        remaining_indices = [i for i in range(len(df)) if i not in completed_indices]
        
        if not remaining_indices:
            print(f"All rows already processed for {model1} vs {model2} - Variant: {variant}. Skipping.")
            return True
            
        # Get prompts for remaining indices
        prompts = []
        for idx in remaining_indices:
            if idx < len(df) and input_column in df.columns:
                prompts.append(df.at[idx, input_column])
        
        print(f"Processing {len(prompts)}/{len(df)} remaining prompts from column '{input_column}'")
        
        # Process using the optimized model processing function
        # Use model-specific settings from util.py
        results = await process_model_prompts(
            m1,  # Model function
            prompts,  # List of prompts
            temperature=m1_temperature,
            prompt_ids=[str(idx) for idx in remaining_indices],
            batch_size=5 if model1 not in ["Llama", "Claude"] else 1,
            progress_desc=f"{model1} → {model2} ({variant})"
        )
        
        # Update dataframe with results
        for i, idx in enumerate(remaining_indices):
            if i < len(results) and idx < len(df):
                # Store the full result
                df.at[idx, output_column] = safe_serialize_result(results[i])
                
                # Extract message content
                df.at[idx, message_column] = extract_message_from_result(results[i])
        
        # Save full results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False, sep='|')
        print(f"Pipeline complete. Results saved to {output_file}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {model1} vs {model2} - Variant: {variant}: {e}")
        # Print more detailed error information for debugging
        import traceback
        traceback.print_exc()
        return False

def load_checkpoints(df, checkpoint_file, output_file, output_column, message_column):
    """
    Load checkpoints and existing output to resume processing.
    
    Args:
        df: DataFrame to update with checkpoint data
        checkpoint_file: Path to checkpoint file
        output_file: Path to output file
        output_column: Name of the output column
        message_column: Name of the message column
        
    Returns:
        list: List of indices that have already been processed
    """
    completed_indices = []
    
    # Check for checkpoint file
    if os.path.exists(checkpoint_file):
        # Load checkpoint data
        try:
            checkpoint_df = pd.read_csv(checkpoint_file, sep='|')
            
            if 'completed' in checkpoint_df.columns:
                # Get indices of completed rows
                completed_rows = checkpoint_df[checkpoint_df['completed'] == True]
                if not completed_rows.empty:
                    completed_indices = completed_rows.index.tolist()
                    print(f"Resuming from checkpoint: {len(completed_indices)}/{len(df)} rows already processed")
                    
                    # Update dataframe with checkpoint data
                    update_df_from_checkpoint(df, checkpoint_df, [output_column, message_column], completed_indices)
        except Exception as e:
            print(f"Warning: Error loading checkpoint file: {e}. Starting from scratch.")
    
    # If output file exists, check it too
    if os.path.exists(output_file):
        try:
            output_df = pd.read_csv(output_file, sep='|')
            if output_column in output_df.columns:
                # Find rows that have output data
                has_output = ~output_df[output_column].isna()
                output_indices = output_df[has_output].index.tolist()
                
                # Update dataframe with output data
                update_df_from_checkpoint(df, output_df, [output_column, message_column], output_indices)
                
                # Combine indices from checkpoint and output
                completed_indices = list(set(completed_indices + output_indices))
                print(f"Found completed rows in output file: {len(completed_indices)}/{len(df)} rows processed")
        except Exception as e:
            print(f"Warning: Error loading output file: {e}")
    
    return completed_indices

def update_df_from_checkpoint(df, source_df, columns, indices):
    """
    Update DataFrame with data from checkpoint or output file.
    
    Args:
        df: Target DataFrame to update
        source_df: Source DataFrame with checkpoint data
        columns: List of column names to copy
        indices: List of row indices to update
    """
    for col in columns:
        if col in source_df.columns:
            for idx in indices:
                if idx < len(df) and idx < len(source_df):
                    df.at[idx, col] = source_df.at[idx, col]

async def main():
    """
    Main function to process all model pairs with controlled concurrency.
    Handles command-line arguments and initializes processing.
    """
    # Define which variants to process
    variants_to_process = []
    
    if args.process_all_m3_variants:
        # Process all variants but only for Grok-Gemini pair
        variants_to_process = list(VARIANT_OUTPUT_PATHS.keys())
        print(f"Will process all variants for Grok-Gemini pair:")
        for variant in variants_to_process:
            input_col = f"step3_input_{variant}"
            print(f"  - {variant} (using input column '{input_col}')")
    else:
        # Just process the specified variant for all model pairs
        variant = args.variant
        variants_to_process = [variant]
        input_col = f"step3_input_{variant}"
        print(f"Will process variant '{variant}' for all model pairs (using input column '{input_col}')")
    
    # Display run configuration
    print(f"{'='*50}")
    print(f"Running with configuration:")
    print(f"  Resume from checkpoint: {args.resume}")
    print(f"  Max concurrent tasks: {MAX_CONCURRENT_TASKS}")
    print(f"  Processing variants: {variants_to_process}")
    print(f"  Checkpoint path: {CHECKPOINT_DIR}")
    print(f"  Checkpoint interval: {CHECKPOINT_INTERVAL} prompts")
    print(f"  Process all M3 variants: {args.process_all_m3_variants}")
    print(f"{'='*50}")

    # Create tasks for all model pairs and variants
    tasks = []
    for variant in variants_to_process:
        for m2 in models:
            for m1 in models:
                if not signal_handler.shutdown_requested:
                    # If processing all variants, only process Grok-Gemini pair
                    if args.process_all_m3_variants:
                        if m1.__name__ == "Grok" and m2.__name__ == "Gemini":
                            tasks.append(process_model_pair(m1, m2, variant))
                    else:
                        tasks.append(process_model_pair(m1, m2, variant))
    
    # Control concurrency with a semaphore
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    
    async def process_with_semaphore(task_func):
        if signal_handler.shutdown_requested:
            return None  # Skip if shutdown requested
        async with semaphore:
            return await task_func
    
    # Process all pairs concurrently with controlled parallelism
    concurrent_tasks = [process_with_semaphore(task) for task in tasks]
    results = await asyncio.gather(*concurrent_tasks)
    
    # Count successes and failures
    successes = sum(1 for result in results if result)
    failures = sum(1 for result in results if not result)
    
    print(f"\n{'='*50}")
    print(f"All model pairs processed!")
    print(f"Successes: {successes}, Failures: {failures}")
    
    if args.process_all_m3_variants:
        print(f"\nVariants processed for Grok-Gemini pair:")
        for variant in variants_to_process:
            output_path = VARIANT_OUTPUT_PATHS[variant]
            summary_dir = f"{output_path}summaries/"
            if os.path.exists(summary_dir) and any(os.path.exists(f"{summary_dir}{f}") for f in os.listdir(summary_dir) if f.startswith("summary_")):
                print(f"  ✓ {variant}")
            else:
                print(f"  ✗ {variant}")
    print(f"{'='*50}")
    
    # Generate combined summaries for each variant
    for variant in variants_to_process:
        print(f"Summary generation for variant '{variant}' will be handled by the merge script")

# Run the main function
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nScript terminated by user. Progress has been saved to checkpoint files.")
        print("To resume, run the script with the --resume flag.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()