#!/usr/bin/env python3
"""
Script to generate stories using story prompts and various LLMs.
Provides utility functions for API handling with parallelized execution.
"""
import pandas as pd
import asyncio
import time
import json
import random
import os
import sys
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.util import *

# Set random seed for reproducibility
random.seed(2025)

# Global parameters
STORY_PROMPT_PREFIX = "please write me a story in exactly 5 sentences:"
INPUT_FILE = "./data/story_seeds/story_seeds_combined_subsampled.csv"
OUTPUT_DIR = "./data/step1"
ERROR_LOG_PATH = f"{OUTPUT_DIR}/error_logs/"

# Define models to use
MODEL_FUNCTIONS = {
    "ChatGPT": ChatGPT,
    "Claude": Claude,
    "Grok": Grok,
    "Gemini": Gemini,
    "Llama": Llama,
    "Deepseek": Deepseek
}

# Initialize text processor
text_processor = TextProcessor()

# Create output and error log directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
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

async def process_model(model_name, df_story_gen, max_concurrent_tasks=1):
    """
    Process a single model to generate stories from prompts.
    
    Args:
        model_name (str): Name of the model to use
        df_story_gen: DataFrame with story prompts
        max_concurrent_tasks (int): Maximum number of concurrent tasks
        
    Returns:
        DataFrame: DataFrame with model outputs or None if failed
    """
    output_path = f"{OUTPUT_DIR}/step1_generate_stories_{model_name}.csv"
    
    # Skip if file already exists
    if os.path.exists(output_path):
        print(f"File {output_path} already exists. Skipping {model_name}.")
        try:
            # Return existing data
            return pd.read_csv(output_path, sep='|')
        except Exception as e:
            print(f"Error reading existing file: {e}")
            return None
    
    # Add a short random delay to stagger API calls
    await asyncio.sleep(random.uniform(0.1, 0.5))
    
    print(f"Processing {model_name}...")
    
    try:
        # Get the model function
        model_func = MODEL_FUNCTIONS.get(model_name)
        if not model_func:
            print(f"Error: Model function {model_name} not found.")
            return None
        
        # Create a copy of the dataframe to avoid modifying the input data
        df_model = df_story_gen.copy()
        prompts = df_model["step1_story_prompt_with_prefix"].tolist()
        
        # Use utility functions for processing prompts with model-specific settings
        temperature = MODEL_TEMPERATURES.get("default", 0.7)
        results = await process_model_prompts(
            model_func,
            prompts,
            temperature=temperature,
            prompt_ids=df_model.index.astype(str).tolist(),
            batch_size=5 if model_name not in ["Llama", "Claude"] else 2,
            progress_desc=f"Generating stories with {model_name}"
        )
        
        # Process results
        df_model["step1_m1_output"] = results
        
        # Extract the first element of the tuple if results are tuples (text, response_object)
        df_model["step1_m1_output_sentence_only"] = [
            extract_message_from_result(result) for result in results
        ]
        
        # Count sentences in the outputs
        df_model["step1_m1_output_sent_count"] = df_model["step1_m1_output_sentence_only"].apply(text_processor.count_sentences)
        
        # Calculate success rate
        error_count = sum(1 for text in df_model["step1_m1_output_sentence_only"] if isinstance(text, str) and text.startswith("Error:"))
        success_rate = (len(df_model) - error_count) / len(df_model) * 100
        print(f"{model_name} success rate: {success_rate:.2f}% ({len(df_model) - error_count}/{len(df_model)})")
        
        # Check for any remaining errors
        error_indices = [i for i, res in enumerate(df_model["step1_m1_output_sentence_only"]) 
                          if isinstance(res, str) and res.startswith("Error:")]
        
        if error_indices:
            print(f"\n⚠️ Warning: {len(error_indices)} responses still have errors after retries")
            
            # Create a summary of error types
            error_codes = {}
            for idx in error_indices:
                response = df_model["step1_m1_output_sentence_only"].iloc[idx]
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
        df_model.to_csv(output_path, index=False, sep='|')
        print(f"Saved results for {model_name} to {output_path}")
        
        # Save evaluation stats
        evaluation = {
            "model": model_name,
            "total_prompts": len(df_model),
            "success_count": len(df_model) - error_count,
            "error_count": error_count,
            "success_rate": success_rate,
            "exact_5_sentences": sum(df_model["step1_m1_output_sent_count"] == 5),
            "sentence_count_distribution": df_model["step1_m1_output_sent_count"].value_counts().to_dict(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save evaluation stats to JSON
        eval_path = f"{OUTPUT_DIR}/step1_eval_{model_name}.json"
        with open(eval_path, "w") as f:
            json.dump(evaluation, f, indent=2)
        
        return df_model
        
    except Exception as e:
        print(f"🚨 ERROR processing {model_name}: {e}")
        # Log the exception
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        error_file = f"{ERROR_LOG_PATH}exception_{model_name}_{timestamp}.log"
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write(f"Exception occurred at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Processing model: {model_name}\n")
            f.write(f"Error: {str(e)}\n")
            
            # Add stack trace for more detailed error information
            import traceback
            f.write("\nStack trace:\n")
            traceback.print_exc(file=f)
            
        print(f"Exception logged to {error_file}")
        return None

async def main(max_concurrent_tasks=6):
    """
    Main function to process story generation for all models concurrently.
    Handles data loading, processing, and saving results.
    
    Args:
        max_concurrent_tasks (int): Maximum number of concurrent tasks to run
    """
    print(f"Starting pipeline with MAX_CONCURRENT_TASKS={max_concurrent_tasks}")
    print(f"Input file: {INPUT_FILE}")
    print(f"Error logs will be saved to: {ERROR_LOG_PATH}")
    
    # Load and prepare data
    try:
        df = pd.read_csv(INPUT_FILE, sep="|")
        print(f"Loaded dataset with {len(df)} rows from {INPUT_FILE}")
    except FileNotFoundError:
        print(f"Input file {INPUT_FILE} not found. Please check the path.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Initialize df_story_gen
    df_story_gen = df.copy()
    
    # Add story prompt prefix
    df_story_gen["step1_story_prompt_with_prefix"] = [
        f"{STORY_PROMPT_PREFIX} {prompt}" for prompt in df_story_gen["prompt"]
    ]
    
    # Get model names from our utility registry
    model_names = list(MODEL_FUNCTIONS.keys())
    
    # Create tasks for all models
    tasks = []
    for model_name in model_names:
        tasks.append(process_model(model_name, df_story_gen))
    
    # Control concurrency with a semaphore to avoid overwhelming the APIs
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    
    async def process_with_semaphore(task_func):
        async with semaphore:
            return await task_func
    
    # Process all models concurrently with controlled parallelism
    concurrent_tasks = [process_with_semaphore(task) for task in tasks]
    results = await asyncio.gather(*concurrent_tasks)
    
    # Filter out None results
    valid_results = [result for result in results if result is not None]
    
    # Count successes and failures
    successes = len(valid_results)
    failures = len(tasks) - successes
    
    print(f"All models processed. Successes: {successes}, Failures: {failures}")
    
    # Check for any error logs
    error_files = [f for f in os.listdir(ERROR_LOG_PATH) if f.endswith('.log')]
    if error_files:
        print(f"\n⚠️ Found {len(error_files)} error log files:")
        for file in sorted(error_files)[:10]:  # Show first 10 files only
            print(f"  - {file}")
        if len(error_files) > 10:
            print(f"  ... and {len(error_files) - 10} more")
        print(f"Check {ERROR_LOG_PATH} for details")
    
    # Create summary DataFrame with stats for each model
    if valid_results:
        summary_data = []
        for model_name in model_names:
            try:
                # Find corresponding result
                result = next((r for r in valid_results if f"step1_generate_stories_{model_name}" in r.columns[0]), None)
                
                if result is not None:
                    df_model = result
                    error_count = sum(1 for text in df_model["step1_m1_output_sentence_only"] if isinstance(text, str) and text.startswith("Error:"))
                    success_rate = (len(df_model) - error_count) / len(df_model) * 100
                    exact_5_count = sum(df_model["step1_m1_output_sent_count"] == 5)
                    exact_5_rate = exact_5_count / len(df_model) * 100
                    
                    summary_data.append({
                        "model": model_name,
                        "total_prompts": len(df_model),
                        "success_count": len(df_model) - error_count,
                        "error_count": error_count,
                        "success_rate": f"{success_rate:.2f}%",
                        "exact_5_sentences": exact_5_count,
                        "exact_5_rate": f"{exact_5_rate:.2f}%"
                    })
                else:
                    # Model failed completely
                    summary_data.append({
                        "model": model_name,
                        "total_prompts": 0,
                        "success_count": 0,
                        "error_count": 0,
                        "success_rate": "0.00%",
                        "exact_5_sentences": 0,
                        "exact_5_rate": "0.00%",
                        "status": "Failed"
                    })
            except Exception as e:
                print(f"Error creating summary for {model_name}: {e}")
        
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            summary_path = f"{OUTPUT_DIR}/step1_summary_all_models.csv"
            df_summary.to_csv(summary_path, index=False, sep='|')
            print(f"Model summary saved to {summary_path}")
    
    print("Pipeline completed!")

# Run the main function
if __name__ == "__main__":
    # Set a higher limit for open files if possible
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, hard), hard))
        print(f"Set file descriptor limit to {min(4096, hard)}")
    except (ImportError, ValueError) as e:
        print(f"Could not adjust file descriptor limit: {e}")
    
    start_time = time.time()
    try:
        # Run with asyncio, default to 6 concurrent tasks (one for each model)
        asyncio.run(main(max_concurrent_tasks=6))
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