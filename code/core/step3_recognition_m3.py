#!/usr/bin/env python3
"""
grok_gemini_claude_prompts.py - Script to read mirror test data where Grok is the story generator (M1), 
Gemini is the marker (M2), and Claude, Gemini, ChatGPT, Llama, or Deepseek to evaluate.
"""
import os
import pandas as pd
import asyncio
import time
import signal
import json
import argparse
from tqdm import tqdm
import sys
# Add the parent directory to path so imports work correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.util import *

# Add command line argument parsing
parser = argparse.ArgumentParser(description='Process mirror test data with different evaluator models')
parser.add_argument('--evaluator', choices=['claude', 'gemini', 'chatgpt', 'llama', 'deepseek', 'all'], default='claude', 
                    help='Choose which model to use as evaluator (claude, gemini, chatgpt, llama, deepseek, or all)')
args = parser.parse_args()

# Set constants and configuration
RANDOM_SEED = 2025
INPUT_DATA_PATH = "./data/step3/input/"
OUTPUT_DATA_PATH_BASE = "./data/step3/output/"
CHECKPOINT_DIR_BASE = "./data/step3/checkpoints/"

# Default batch size, will be overridden for specific models
BATCH_SIZE = 5

# Function to process a single evaluator
async def process_evaluator(evaluator_name):
    """Process a single evaluator configuration"""
    
    if evaluator_name == 'claude':
        EVALUATOR = "Claude"
        OUTPUT_DATA_PATH = f"{OUTPUT_DATA_PATH_BASE}grok_gemini_claude/"
        CHECKPOINT_DIR = f"{CHECKPOINT_DIR_BASE}grok_gemini_claude/"
        BATCH_SIZE = 1  # Set to 1 for Claude to prevent rate limiting
        model_temperature = 0.7
        model_function = Claude
    elif evaluator_name == 'gemini':
        EVALUATOR = "Gemini"
        OUTPUT_DATA_PATH = f"{OUTPUT_DATA_PATH_BASE}grok_gemini_gemini/"
        CHECKPOINT_DIR = f"{CHECKPOINT_DIR_BASE}grok_gemini_gemini/"
        BATCH_SIZE = 5  # Can use larger batch size for Gemini
        model_temperature = 0.7
        model_function = Gemini
    elif evaluator_name == 'chatgpt':
        EVALUATOR = "ChatGPT"
        OUTPUT_DATA_PATH = f"{OUTPUT_DATA_PATH_BASE}grok_gemini_chatgpt/"
        CHECKPOINT_DIR = f"{CHECKPOINT_DIR_BASE}grok_gemini_chatgpt/"
        BATCH_SIZE = 5
        model_temperature = 0.7
        model_function = ChatGPT
    elif evaluator_name == 'llama':
        EVALUATOR = "Llama"
        OUTPUT_DATA_PATH = f"{OUTPUT_DATA_PATH_BASE}grok_gemini_llama/"
        CHECKPOINT_DIR = f"{CHECKPOINT_DIR_BASE}grok_gemini_llama/"
        BATCH_SIZE = 5
        model_temperature = 0.7
        model_function = Llama
    elif evaluator_name == 'deepseek':
        EVALUATOR = "Deepseek"
        OUTPUT_DATA_PATH = f"{OUTPUT_DATA_PATH_BASE}grok_gemini_deepseek/"
        CHECKPOINT_DIR = f"{CHECKPOINT_DIR_BASE}grok_gemini_deepseek/"
        BATCH_SIZE = 5
        model_temperature = 0.7
        model_function = Deepseek
    else:
        print(f"Unknown evaluator: {evaluator_name}")
        return False
    
    # Create file paths for this specific combination
    INPUT_FILE = f"{INPUT_DATA_PATH}mirror_test_results_Grok_Gemini_processed.csv"
    OUTPUT_FILE = f"{OUTPUT_DATA_PATH}mirror_test_results_Grok_Gemini_{EVALUATOR}.csv"
    CHECKPOINT_FILE = f"{CHECKPOINT_DIR}mirror_test_results_Grok_Gemini_{EVALUATOR}_checkpoint.csv"

    # Create necessary directories
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print(f"Processing Step 3 with {EVALUATOR} as evaluator for Grok stories marked by Gemini")
    
    # Skip if output file already exists
    if os.path.exists(OUTPUT_FILE):
        print(f"File {OUTPUT_FILE} already exists. Skipping this triplet.")
        return True
    
    # Skip if input file doesn't exist
    if not os.path.exists(INPUT_FILE):
        print(f"Input file {INPUT_FILE} does not exist. Skipping.")
        return False
    
    try:
        # Load the processed data from step 2
        df = pd.read_csv(INPUT_FILE, sep='|')
        print(f"Loaded {len(df)} records from {INPUT_FILE}")
        
        # Check if the input column exists
        if "step3_input_standard" not in df.columns:
            print(f"Error: Column 'step3_input_standard' not found in the input file. Available columns: {df.columns.tolist()}")
            print(f"Skipping due to missing required column.")
            return False
        
        # Define columns for evaluator output
        output_column = f"step3_{evaluator_name}_output"
        message_column = f"step3_{evaluator_name}_output_message_only"
        
        # Check for existing checkpoint
        completed_indices = []
        
        if os.path.exists(CHECKPOINT_FILE):
            try:
                checkpoint_df = pd.read_csv(CHECKPOINT_FILE, sep='|')
                
                if 'completed' in checkpoint_df.columns:
                    # Get indices of completed rows
                    completed_rows = checkpoint_df[checkpoint_df['completed'] == True]
                    if not completed_rows.empty:
                        completed_indices = completed_rows.index.tolist()
                        print(f"Resuming from checkpoint: {len(completed_indices)}/{len(df)} rows already processed")
                        
                        # Update dataframe with checkpoint data
                        for col in [output_column, message_column]:
                            if col in checkpoint_df.columns:
                                for idx in completed_indices:
                                    if idx < len(df) and idx < len(checkpoint_df):
                                        df.at[idx, col] = checkpoint_df.at[idx, col]
            except Exception as e:
                print(f"Warning: Error loading checkpoint file: {e}. Starting from scratch.")
        
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
            print(f"All rows already processed. Skipping.")
            return True
            
        # Get prompts for remaining indices
        prompts = []
        for idx in remaining_indices:
            if idx < len(df):
                prompts.append(df.at[idx, "step3_input_standard"])

        print(f"Processing {len(prompts)}/{len(df)} remaining prompts")
        
        # Process using selected model
        results = await process_model_prompts(
            model_function,  # Selected model function
            prompts,  # List of prompts
            temperature=model_temperature,
            prompt_ids=[str(idx) for idx in remaining_indices],
            batch_size=BATCH_SIZE,  # Use appropriate batch size
            progress_desc=f"Grok vs Gemini → {EVALUATOR}"
        )
        
        # Update dataframe with results
        for i, idx in enumerate(remaining_indices):
            if i < len(results) and idx < len(df):
                # Store the full result
                df.at[idx, output_column] = safe_serialize_result(results[i])
                
                # Extract message content
                df.at[idx, message_column] = extract_message_from_result(results[i])
        
        # Extract numeric value from the output message using pandas methods
        output_int_column = f"step3_{evaluator_name}_output_int"
        df[output_int_column] = pd.to_numeric(
            df[message_column].str.extract(r'(\d+)').iloc[:, 0],
            errors='coerce'
        ).fillna(0).astype(int)
        
        # Calculate accuracy of evaluator compared to expected result
        # Handle potential NA values by explicitly filtering them out
        valid_rows = df.dropna(subset=['step2_random_sent_num', output_int_column])
        if len(valid_rows) > 0:
            matches_expected = sum(valid_rows['step2_random_sent_num'] == valid_rows[output_int_column])
            accuracy = matches_expected / len(valid_rows)
        else:
            accuracy = 0
            print(f"Warning: No valid response pairs found for accuracy calculation")
            
        print(f"Accuracy of {EVALUATOR} evaluating Grok vs Gemini: {accuracy:.2%}")
        
        # Log accuracy in the DataFrame for reference
        accuracy_column = f"step3_{evaluator_name}_accuracy"
        df[accuracy_column] = accuracy
        
        # Save results
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False, sep='|')
        print(f"Results saved to {OUTPUT_FILE}")
        
        # Create summary
        summary = {
            "model1": "Grok",
            "model2": "Gemini",
            "model3": EVALUATOR,
            "total_samples": len(df),
            "valid_samples": len(valid_rows),
            "accuracy": accuracy,
            "is_self_recognition": False,
            "is_self_evaluation": EVALUATOR == "Gemini",  # True if Gemini is evaluating itself
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save summary to JSON
        summary_dir = f"{OUTPUT_DATA_PATH}summaries/"
        os.makedirs(summary_dir, exist_ok=True)
        summary_file = f"{summary_dir}summary_Grok_Gemini_{EVALUATOR}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"Error processing Grok vs Gemini → {EVALUATOR}: {e}")
        # Print more detailed error information for debugging
        import traceback
        traceback.print_exc()
        return False

# Add signal handler for graceful shutdown
def signal_handler(sig, frame):
    print('\nReceived interrupt signal. Completing current batch before exiting...')
    # Set a flag to tell asyncio to stop scheduling new tasks
    signal_handler.shutdown_requested = True

# Initialize the flag
signal_handler.shutdown_requested = False

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# Main async function
async def main():
    """
    Main function to process evaluators based on command line argument
    """
    # Display run configuration
    print(f"{'='*50}")
    print(f"Running with configuration:")
    print(f"  Evaluator: {args.evaluator}")
    print(f"{'='*50}")
    
    if args.evaluator == 'all':
        # Process all evaluators sequentially
        evaluators = ['claude', 'gemini', 'chatgpt', 'llama', 'deepseek']
        results = {}
        
        for evaluator in evaluators:
            print(f"\nProcessing {evaluator.upper()} as evaluator...")
            success = await process_evaluator(evaluator)
            results[evaluator] = success
            
        # Print summary of results
        print("\nResults summary:")
        for evaluator, success in results.items():
            status = "Successfully completed" if success else "Failed or skipped"
            print(f"  {evaluator.upper()}: {status}")
    else:
        # Process single evaluator
        success = await process_evaluator(args.evaluator)
        
        if success:
            print("Processing completed successfully!")
        else:
            print("Processing failed or was skipped.")

# Run the main function
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nScript terminated by user. Progress has been saved to checkpoint files.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()