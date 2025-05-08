#!/usr/bin/env python3
"""
Step 2: Adds a mark to one sentence in the story by having a second model rewrite it.
Provides utility functions for API handling and error management.
"""
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import asyncio
import time
import re
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.util import *

# Set constants and configuration
RANDOM_SEED = 2025
SENTENCE_COUNT_TARGET = 5
OUTPUT_DATA_PATH = "./data/step2/"
OUTPUT_FILE_TEMPLATE = f"{OUTPUT_DATA_PATH}output/mirror_test_results_{{}}_{{}}.csv"
INPUT_FILE = f"{OUTPUT_DATA_PATH}input/mirror_test_results_add_random_result_exactly_{SENTENCE_COUNT_TARGET}_sentences_combined.csv"
ERROR_LOG_PATH = f"{OUTPUT_DATA_PATH}error_logs/"

# Set random seed for reproducibility
random.seed(RANDOM_SEED)

# Define models to use
MODEL_FUNCTIONS = {
    "ChatGPT": ChatGPT,
    "Claude": Claude,
    "Grok": Grok,
    "Gemini": Gemini,
    "Llama": Llama,
    "Deepseek": Deepseek
}

# Create output and error log directories if they don't exist
os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
os.makedirs(f"{OUTPUT_DATA_PATH}result/", exist_ok=True)
os.makedirs(ERROR_LOG_PATH, exist_ok=True)

# Function to log errors
def log_error(model_name, prompt_id, error_code, prompt, response):
    """
    Log API errors to file for debugging.
    
    Args:
        model_name (str): Name of the model
        prompt_id (str): Identifier for the prompt
        error_code (str): Error code or description
        prompt (str): The input prompt
        response (str): The error response
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    error_file = f"{ERROR_LOG_PATH}{model_name}_errors_{timestamp}.log"
    
    with open(error_file, 'a', encoding='utf-8') as f:
        f.write(f"===== ERROR DETAILS =====\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Prompt ID: {prompt_id}\n")
        f.write(f"Error Code: {error_code}\n")
        f.write(f"Prompt:\n{prompt}\n\n")
        f.write(f"Response:\n{response}\n")
        f.write(f"========================\n\n")
    
    print(f"Error logged to {error_file}")

async def process_model_pair(model1_name, model2_name, max_concurrent_tasks=1):
    """
    Process a pair of models: use model1's stories and have model2 rewrite a random sentence.
    
    Args:
        model1_name (str): Name of the first model (story source)
        model2_name (str): Name of the second model (sentence modifier)
        max_concurrent_tasks (int): Maximum number of concurrent tasks to run
        
    Returns:
        bool: True if processing succeeded, False otherwise
    """
    print(f"Processing: {model1_name} vs {model2_name}")
    
    # Define output file path
    output_file = OUTPUT_FILE_TEMPLATE.format(model1_name, model2_name)
    
    # Skip if file already exists
    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping this pair.")
        return True
        
    # Add a short random delay to stagger API calls when multiple pairs start simultaneously
    await asyncio.sleep(random.uniform(0.1, 0.5))
    
    try:
        # Load the combined data
        df_all = pd.read_csv(INPUT_FILE, sep='|')
        
        # Filter data for the first model
        df = df_all[df_all.model == model1_name].copy()
        
        # Log statistics
        avg_length = np.mean([len(x) for x in df['step1_m1_output_sentence_only']])
        print(f"Average length of step1_m1_output_sentence_only: {avg_length:.2f}")
        print(f"Processing {len(df)} stories for {model1_name} → {model2_name}")
        
        # Step 2: Generate replacement sentences
        prompts = df["step2_marked_text_input_nth_sentence"].tolist()
        prompt_ids = df.index.astype(str).tolist()
        
        # Get the model function for the second model
        model2_func = MODEL_FUNCTIONS.get(model2_name)
        if not model2_func:
            print(f"Error: Model function {model2_name} not found.")
            return False
        
        # Use the centralized function for processing prompts
        temperature = MODEL_TEMPERATURES.get("default", 0.7)
        
        # Process with model-specific optimizations
        results = await process_model_prompts(
            model2_func,
            prompts,
            temperature=temperature,
            prompt_ids=prompt_ids,
            batch_size=5 if model2_name not in ["Llama", "Claude"] else 1,
            progress_desc=f"{model1_name} → {model2_name}"
        )
        
        # Add results to DataFrame
        df["step2_output_nth_sentence"] = results
        
        # Extract message content
        df["step2_output_nth_sentence_message_only"] = [
            extract_message_from_result(result) for result in results
        ]
        
        # Check for any remaining errors in the final results
        error_indices = [i for i, res in enumerate(df["step2_output_nth_sentence_message_only"]) 
                          if isinstance(res, str) and (res.startswith("Error") or "Failed after" in res)]
        
        if error_indices:
            print(f"\n⚠️ Warning: {len(error_indices)} responses still have errors after retries")
            
            # Create a summary of error types
            error_codes = {}
            for idx in error_indices:
                response = df["step2_output_nth_sentence_message_only"].iloc[idx]
                error_code = "unknown"
                if "429" in response:
                    error_code = "429"
                elif "rate limit" in response.lower():
                    error_code = "rate_limit"
                elif "error" in response.lower():
                    error_code = "api_error"
                
                if error_code not in error_codes:
                    error_codes[error_code] = 0
                error_codes[error_code] += 1
            
            print("Error code summary:")
            for code, count in error_codes.items():
                print(f"  - Code {code}: {count} occurrences")
        
        # Save results
        df.to_csv(output_file, index=False, sep='|')
        print(f"Pipeline complete. Results saved to {output_file}")
        return True
        
    except Exception as e:
        print(f"🚨 ERROR processing {model1_name} vs {model2_name}: {e}")
        # Log the exception
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        error_file = f"{ERROR_LOG_PATH}exception_{model1_name}_{model2_name}_{timestamp}.log"
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write(f"Exception occurred at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Processing models: {model1_name} vs {model2_name}\n")
            f.write(f"Error: {str(e)}\n")
            
            # Add stack trace for more detailed error information
            import traceback
            f.write("\nStack trace:\n")
            traceback.print_exc(file=f)
            
        print(f"Exception logged to {error_file}")
        return False

async def main(max_concurrent_tasks=2):
    """
    Main function to process all model pairs with controlled concurrency.
    
    Args:
        max_concurrent_tasks (int): Maximum number of concurrent tasks to run
    """
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    os.makedirs(f"{OUTPUT_DATA_PATH}input/", exist_ok=True)
    os.makedirs(f"{OUTPUT_DATA_PATH}output/", exist_ok=True)
    os.makedirs(f"{OUTPUT_DATA_PATH}result/", exist_ok=True)
    os.makedirs(ERROR_LOG_PATH, exist_ok=True)

    print(f"Starting pipeline with MAX_CONCURRENT_TASKS={max_concurrent_tasks}")
    print(f"Input file: {INPUT_FILE}")
    print(f"Error logs will be saved to: {ERROR_LOG_PATH}")
    
    # Get model names from our utility registry
    model_names = list(MODEL_FUNCTIONS.keys())
    
    # Validate input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file {INPUT_FILE} does not exist.")
        return
    
    # Create tasks for all model pairs or specific pairs as needed
    tasks = []
    for model1_name in model_names:
        for model2_name in model_names:
            # You can comment this out to process only specific pairs
            tasks.append(process_model_pair(model1_name, model2_name))
    
    # Control concurrency with a semaphore to avoid overwhelming the APIs
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    
    async def process_with_semaphore(task_func):
        async with semaphore:
            return await task_func
    
    # Process all pairs concurrently with controlled parallelism
    concurrent_tasks = [process_with_semaphore(task) for task in tasks]
    results = await asyncio.gather(*concurrent_tasks)
    
    # Count successes and failures
    successes = sum(1 for result in results if result)
    failures = sum(1 for result in results if not result)
    
    print(f"All model pairs processed. Successes: {successes}, Failures: {failures}")
    
    # Check for any error logs
    error_files = [f for f in os.listdir(ERROR_LOG_PATH) if f.endswith('.log')]
    if error_files:
        print(f"\n⚠️ Found {len(error_files)} error log files:")
        for file in sorted(error_files)[:10]:  # Show first 10 files only
            print(f"  - {file}")
        if len(error_files) > 10:
            print(f"  ... and {len(error_files) - 10} more")
        print(f"Check {ERROR_LOG_PATH} for details")
    
    # Create a summary of results
    create_summary()

def create_summary():
    """Create a summary of all processed pairs"""
    result_dir = f"{OUTPUT_DATA_PATH}result/"
    files = [f for f in os.listdir(result_dir) if f.endswith('.csv')]
    
    summary_data = []
    for file in files:
        # Extract model names from filename
        match = re.match(r'mirror_test_results_(.+)_(.+)\.csv', file)
        if match:
            model1, model2 = match.groups()
            
            # Get file stats
            file_path = os.path.join(result_dir, file)
            try:
                df = pd.read_csv(file_path, sep='|')
                row_count = len(df)
                error_count = 0
                
                # Check for error rows if the column exists
                if "step2_output_nth_sentence_message_only" in df.columns:
                    error_count = sum(1 for text in df["step2_output_nth_sentence_message_only"] 
                                     if isinstance(text, str) and (text.startswith("Error") or "Failed after" in text))
                
                summary_data.append({
                    "source_model": model1,
                    "target_model": model2,
                    "row_count": row_count,
                    "error_count": error_count,
                    "success_rate": f"{(row_count - error_count) / row_count * 100:.2f}%" if row_count > 0 else "N/A"
                })
            except Exception as e:
                print(f"Error reading {file}: {e}")
    
    if summary_data:
        # Create summary dataframe
        df_summary = pd.DataFrame(summary_data)
        summary_path = f"{OUTPUT_DATA_PATH}step2_summary.csv"
        df_summary.to_csv(summary_path, index=False, sep='|')
        print(f"Summary created and saved to {summary_path}")

# Run the main function
if __name__ == "__main__":
    start_time = time.time()
    try:
        # Default to 6 concurrent tasks, adjust based on your environment
        asyncio.run(main(max_concurrent_tasks=1))
    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user")
    except Exception as e:
        print(f"\n🚨 CRITICAL ERROR: {str(e)}")
        # Print detailed error information
        import traceback
        traceback.print_exc()
    finally:
        elapsed_time = time.time() - start_time
        print(f"Total execution time: {elapsed_time:.2f} seconds")