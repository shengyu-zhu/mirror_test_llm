#!/usr/bin/env python3
"""
Script to generate story prompts using multiple LLMs in parallel.
Provides utility functions for API handling.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.util import *
import json
import time
import asyncio
import pandas as pd

# Parameter settings
NUM_ITERATIONS = 20  # Number of iterations to run
BASE_PROMPT = "provide me 50 prompts to generate short stories in a python list"
OUTPUT_DIR = "./data/story_seeds/raw_responses"
MODELS_OUTPUT_DIR = f"{OUTPUT_DIR}/models"

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)

async def process_iteration(iteration, prompt, temperature=0.7):
    """
    Process a single iteration with all models in parallel and save raw responses.
    
    Args:
        iteration (int): Current iteration number
        prompt (str): The prompt to send to all models
        temperature (float): Temperature setting for generation
        
    Returns:
        dict: Dictionary containing model responses
    """
    print(f"\n\n{'='*50}")
    print(f"ITERATION {iteration} OF {NUM_ITERATIONS}")
    print(f"{'='*50}\n")
    
    # Define all models to use
    models = [ChatGPT, Claude, Grok, Gemini, Llama, Deepseek]
    model_names = [model.__name__ for model in models]
    
    # Create prompts list (same prompt for all models)
    prompts = [prompt] * len(models)
    
    # Process all models in parallel with optimized handling
    results = []
    for i, model_func in enumerate(models):
        # Use the process_model_prompts utility function with model-specific optimizations
        model_result = await process_model_prompts(
            model_func, 
            [prompt],  # Single prompt per model
            temperature=temperature,
            progress_desc=f"Processing {model_names[i]}"
        )
        results.append(model_result[0])  # Get the first (only) result
        
    # Extract text responses from results
    responses = [extract_message_from_result(result) for result in results]
    
    # Print results
    for model_name, response in zip(model_names, responses):
        print(f"{model_name}:")
        print(response[:500] + "..." if len(response) > 500 else response)
        print("\n" + "-"*40 + "\n")
    
    # Save raw responses to JSON file
    responses_dict = {model: resp for model, resp in zip(model_names, responses)}
    responses_dict["iteration"] = iteration
    responses_dict["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Save to JSON file for this iteration
    output_file = f"{OUTPUT_DIR}/responses_iteration_{iteration}.json"
    with open(output_file, "w") as f:
        json.dump(responses_dict, f, indent=2)
    
    # Save individual model responses to separate JSON files
    for model_name, response in zip(model_names, responses):
        model_data = {
            "model": model_name,
            "response": response,
            "iteration": iteration,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": temperature,
            "prompt": prompt
        }
        model_output_file = f"{MODELS_OUTPUT_DIR}/{model_name}_iteration_{iteration}.json"
        with open(model_output_file, "w") as f:
            json.dump(model_data, f, indent=2)
        print(f"Model {model_name} response saved to {model_output_file}")
    
    print(f"Raw responses saved to {output_file}")
    
    return responses_dict

async def main():
    """
    Main function to generate story prompts across multiple iterations and models.
    Handles data collection, processing, and saving results.
    """
    # Create dictionaries to store all responses by model
    model_names = ["ChatGPT", "Claude", "Grok", "Gemini", "Llama", "Deepseek"]
    all_model_responses = {model: [] for model in model_names}
    all_responses = []
    
    # Define specific temperature for consistency
    temperature = MODEL_TEMPERATURES.get("default", 0.7)
    
    for iteration in range(1, NUM_ITERATIONS + 1):
        try:
            responses_dict = await process_iteration(iteration, BASE_PROMPT, temperature)
            all_responses.append(responses_dict)
            
            # Add each model's response to its collection
            for model in model_names:
                model_response = {
                    "iteration": iteration,
                    "response": responses_dict[model],
                    "timestamp": responses_dict["timestamp"],
                    "temperature": temperature,
                    "prompt": BASE_PROMPT
                }
                all_model_responses[model].append(model_response)
            
            # Add a small delay between iterations to avoid rate limits
            if iteration < NUM_ITERATIONS:
                print(f"\nWaiting 10 seconds before starting next iteration...")
                await asyncio.sleep(10)
        except Exception as e:
            print(f"Error processing iteration {iteration}: {e}")
            # Save checkpoint in case of error
            checkpoint_file = f"{OUTPUT_DIR}/checkpoint_iteration_{iteration}.json"
            with open(checkpoint_file, "w") as f:
                json.dump(all_responses, f, indent=2)
            print(f"Checkpoint saved to {checkpoint_file}")

    # Save combined responses
    with open(f"{OUTPUT_DIR}/all_responses.json", "w") as f:
        json.dump(all_responses, f, indent=2)
    
    # Save aggregated responses for each model
    for model in model_names:
        model_file = f"{MODELS_OUTPUT_DIR}/{model}_all_iterations.json"
        with open(model_file, "w") as f:
            json.dump(all_model_responses[model], f, indent=2)
        print(f"All {model} responses saved to {model_file}")
    
    print("\nAll raw responses collected and saved!")
    
    # Process the prompts to extract and save as CSV
    process_all_prompts(all_responses)

def process_all_prompts(all_responses):
    """
    Process the collected responses to extract and organize story prompts.
    
    Args:
        all_responses (list): List of response dictionaries from all iterations
    """
    all_prompts = []
    for iteration_data in all_responses:
        iteration = iteration_data["iteration"]
        timestamp = iteration_data["timestamp"]
        
        for model in ["ChatGPT", "Claude", "Grok", "Gemini", "Llama", "Deepseek"]:
            response = iteration_data[model]
            
            # Extract prompts from the response
            try:
                # Try to find and extract a Python list in the text
                import re
                list_pattern = r'\[.*?\]'
                match = re.search(list_pattern, response, re.DOTALL)
                
                if match:
                    list_text = match.group()
                    
                    # Safely evaluate the list
                    import ast
                    try:
                        prompts_list = ast.literal_eval(list_text)
                        if isinstance(prompts_list, list):
                            # Add metadata to each prompt
                            for i, prompt in enumerate(prompts_list):
                                all_prompts.append({
                                    'model': model,
                                    'iteration': iteration,
                                    'prompt_index': i + 1,
                                    'timestamp': timestamp,
                                    'prompt': prompt
                                })
                    except (SyntaxError, ValueError) as e:
                        print(f"Error parsing list from {model} response: {e}")
                else:
                    print(f"No list pattern found in {model} response")
            except Exception as e:
                print(f"Error processing {model} response: {e}")
    
    # Convert to DataFrame and save
    if all_prompts:
        df_prompts = pd.DataFrame(all_prompts)
        os.makedirs("./data/story_seeds", exist_ok=True)
        df_prompts.to_csv("./data/story_seeds/story_seeds_combined.csv", index=False, sep='|')
        print(f"Extracted {len(df_prompts)} prompts and saved to story_seeds_combined.csv")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())