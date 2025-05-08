import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union, List, Tuple, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from functools import partial
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.util import *

# Set constants and configuration
RANDOM_SEED = 2025
INPUT_DATA_PATH = "./data/step3/output/"
OUTPUT_DATA_PATH = "./data/step4/output/"
INPUT_FILE_TEMPLATE = f"{OUTPUT_DATA_PATH}mirror_test_results_{{}}_{{}}.csv"
OUTPUT_FILE_TEMPLATE = f"{OUTPUT_DATA_PATH}mirror_test_results_{{}}_{{}}_explanation.csv"
MAX_CONCURRENT_TASKS = 6
# Specific limit for Llama model to avoid concurrency conflicts
LLAMA_CONCURRENCY_LIMIT = 1  # Only process one Llama request at a time

# Set random seed for reproducibility
random.seed(RANDOM_SEED)

# Define models to compare
models = [ChatGPT, Claude, Grok, Gemini, Llama, Deepseek]
text_processor = TextProcessor()

# Global parameters
time_sleep_buffer = 0.1  # Buffer time between API calls
m1_temperature = 0.7  # Set temperature for model responses

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

# Create a dedicated semaphore for Llama API calls
llama_semaphore = asyncio.Semaphore(LLAMA_CONCURRENCY_LIMIT)

# Process a single model pair
async def process_model_pair(m1, m2):
    model1 = m1.__name__
    model2 = m2.__name__
    print(f"Processing Step 3: {model1} vs {model2}")
    
    # Define input and output file paths
    input_file = INPUT_FILE_TEMPLATE.format(model1, model2)
    output_file = OUTPUT_FILE_TEMPLATE.format(model1, model2)
    
    # Skip if output file already exists
    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping this pair.")
        return
    
    # Skip if input file doesn't exist
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist. Skipping this pair.")
        return
        
    # Add a short random delay to stagger API calls when multiple pairs start simultaneously
    await asyncio.sleep(random.uniform(0.1, 0.5))
    
    try:
        # Load the data from step 3
        df = pd.read_csv(input_file, sep='|')

        # Log statistics
        print(f"Loaded {len(df)} records from {input_file}")
        
        # Step 4: Ask model reason for its prior choice of strange sentence
        df["step4_input"] = [
            f"I previously asked you, 'what's the part of the paragraph you think is strange, only answer the sentence: {text1}. You answered: {text2}. Now, please provide a reason for your choice by returning just one of the following concepts that best describes your reasoning: Awkward Word Choice; Strange or Forced Idioms; Mixed/Forced Metaphors; Overly Florid / Purple Prose; Illogical or Nonsensical Imagery; Inconsistent Tone; Overly Melodramatic; Inconsistent Diction; Inconsistent Sentence Structure; Confusing or Vague Construction; Other."
            for text1, text2 in zip(
                df["step2_output"], df["step3_output_message_only"]
            )
        ]
        prompts = df["step4_input"].tolist()
        print(f"Processing {len(prompts)} prompts")
        
        # Process based on model type
        results = []
        is_llama = model1 == "Llama"
        
        # Special handling for Llama and Claude
        if model1 in ["Llama", "Claude"]:
            for prompt in tqdm(prompts, desc=f"{model1} → {model2}"):
                # Pass is_llama flag for Llama model
                result = await async_model_call(m1, prompt, m1_temperature, is_llama=is_llama, llama_semaphore=llama_semaphore)
                results.append(result)
                # Add specific wait times based on model
                wait_time = 0.5 if model1 == "Llama" else 0.3
                await asyncio.sleep(wait_time)
        else:
            # Process in batches for better throughput and progress tracking
            batch_size = 5  # Adjust based on API rate limits
            
            for i in tqdm(range(0, len(prompts), batch_size), desc=f"{model1} → {model2}"):
                batch_prompts = prompts[i:i+batch_size]
                batch_results = await process_batch(m1, batch_prompts, m1_temperature, is_llama=False)
                results.extend(batch_results)
                
                # Add a small delay between batches to avoid rate limiting
                await asyncio.sleep(time_sleep_buffer)
        
        # Add results to DataFrame
        df["step4_output"] = results
        
        # Extract message content
        df["step4_output_message_only"] = [extract_message_from_result(result) for result in results]
        
        # Save results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False, sep='|')
        print(f"Pipeline complete. Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error processing {model1} vs {model2}: {e}")

# Main async function
async def main():
    # Create tasks for all model pairs
    tasks = []
    for m2 in models:
        for m1 in models:
            tasks.append(process_model_pair(m1, m2))
    
    # Control concurrency with a semaphore to avoid overwhelming the APIs
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    
    async def process_with_semaphore(task_func):
        async with semaphore:
            return await task_func
    
    # Process all non-Llama pairs concurrently with controlled parallelism
    concurrent_tasks = [process_with_semaphore(task) for task in tasks]
    await asyncio.gather(*concurrent_tasks)

    print("All model pairs processed successfully for Step 3!")

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())