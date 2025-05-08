import pandas as pd
import re
import csv
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.util import *

import numpy as np
import random
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize

# Configuration parameters
PER_MODEL_COUNT = 200

# Set a fixed seed for reproducibility
random.seed(2025)

def extract_prompts_from_string(content):
    """
    Extract all prompts from a string containing a Python list.
    Works with different formats including Claude's output.
    """
    # Try to find a list assignment pattern (either story_prompts or prompts)
    match = re.search(r'(?:story_prompts|prompts)\s*=\s*\[([\s\S]*?)\]', content)
    
    if not match:
        # If no list assignment found, try to extract anything that looks like a list
        match = re.search(r'\[\s*([\s\S]*?)\s*\]', content)
        
    if not match:
        return []
        
    prompts_text = match.group(1)
    
    # Extract each quoted string
    prompt_pattern = r'"([^"\\]*(?:\\.[^"\\]*)*)"'
    prompts = re.findall(prompt_pattern, prompts_text)
    
    # If no prompts found with double quotes, try single quotes
    if not prompts:
        prompt_pattern = r"'([^'\\]*(?:\\.[^'\\]*)*)'"
        prompts = re.findall(prompt_pattern, prompts_text)
    
    # Clean up each prompt (handle escaped quotes, etc.)
    cleaned_prompts = []
    for prompt in prompts:
        # Handle escaped quotes
        cleaned = prompt.replace('\\"', '"').replace("\\'", "'")
        if cleaned:
            cleaned_prompts.append(cleaned)
    
    return cleaned_prompts

def remove_quotes(text):
    """Remove leading and trailing quotes from a string"""
    if isinstance(text, str):
        # Check if the string starts and ends with quotes
        if text.startswith('"') and text.endswith('"'):
            return text[1:-1]
    return text

def process_individual_model_file(file_path):
    """Process a single model response file and return a DataFrame"""
    # Load JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract model name and iteration number
    model_name = data["model"]
    iteration = data["iteration"]
    response = data["response"]
    
    # Extract prompts from the response
    prompts = extract_prompts_from_string(response)
    
    # Create data list for DataFrame
    df_data = []
    
    # Add each prompt to the data list
    for i, prompt in enumerate(prompts):
        df_data.append({
            'prompt': prompt,
            'model': model_name,
            'order': i + 1,  # 1-indexed position
            'iteration': iteration
        })
    
    # Create DataFrame
    df = pd.DataFrame(df_data)
    
    # Apply the function to the 'prompt' column
    if not df.empty:
        df['prompt'] = df['prompt'].apply(remove_quotes)
    
    return df

def analyze_prompt_lengths(df):
    """Calculate and analyze prompt lengths"""
    # Calculate prompt lengths
    df['prompt_length'] = df['prompt'].str.len()
    
    # Group by model and iteration to get length statistics
    length_stats = df.groupby(['model', 'iteration'])['prompt_length'].agg([
        'count', 'mean', 'min', 'max', 'sum', 
        lambda x: np.percentile(x, 25),  # 25th percentile
        lambda x: np.percentile(x, 50),  # median
        lambda x: np.percentile(x, 75)   # 75th percentile
    ]).reset_index()
    
    # Rename columns for clarity
    length_stats = length_stats.rename(columns={
        'count': 'num_prompts',
        'mean': 'avg_length',
        'min': 'min_length',
        'max': 'max_length', 
        'sum': 'total_chars',
        '<lambda_0>': 'p25_length',
        '<lambda_1>': 'median_length',
        '<lambda_2>': 'p75_length'
    })
    
    # Round the average and percentiles to 2 decimal places
    length_stats['avg_length'] = length_stats['avg_length'].round(2)
    length_stats['p25_length'] = length_stats['p25_length'].round(2)
    length_stats['median_length'] = length_stats['median_length'].round(2)
    length_stats['p75_length'] = length_stats['p75_length'].round(2)
    
    return length_stats

def select_random_prompts(all_data, per_model_count=PER_MODEL_COUNT):
    """
    Select random prompts from the processed data and prepare them for story generation.
    
    Args:
        all_data (pandas.DataFrame): The DataFrame containing all prompts
        per_model_count (int): Number of prompts to select per model
    
    Returns:
        pandas.DataFrame: A DataFrame containing selected prompts prepared for story generation
    """
    # Create output directory if it doesn't exist
    os.makedirs("./data/story_generation", exist_ok=True)
    
    # Rename model column to indicate it's the source of the original prompt
    all_data = all_data.rename(columns={'model': 'model_of_original_prompt'})
    
    # Display information about the dataset
    print("\n" + "="*50)
    print("DATASET INFORMATION FOR PROMPT SELECTION:")
    print("="*50)
    print(f"Total prompts: {len(all_data)}")
    print(f"Unique models: {all_data['model_of_original_prompt'].nunique()}")
    print(f"Unique iterations: {all_data['iteration'].nunique()}")
    
    # Initialize list to hold selected prompts
    selected_prompts_list = []
    
    # Group by model
    model_groups = all_data.groupby('model_of_original_prompt')
    
    # For each model, select random prompts
    for model, model_data in model_groups:
        print(f"\nProcessing model: {model}")
        print(f"Total prompts available: {len(model_data)}")
        
        # If there are fewer prompts than requested, take all of them
        if len(model_data) <= per_model_count:
            selected = model_data
            print(f"Selected all {len(selected)} prompts (fewer than requested {per_model_count})")
        else:
            # Randomly sample prompts from this model
            selected = model_data.sample(per_model_count, random_state=2025)
            print(f"Randomly selected {len(selected)} prompts")
        
        # Add to our collection
        selected_prompts_list.append(selected)
    
    # Concatenate all selected prompts
    df_story_gen = pd.concat(selected_prompts_list).reset_index(drop=True)
    
    # Print information about the selected prompts
    print("\n" + "="*50)
    print("SELECTED PROMPTS INFORMATION:")
    print("="*50)
    print(f"Total number of selected prompts: {len(df_story_gen)}")
    print(f"Models represented: {df_story_gen['model_of_original_prompt'].unique()}")
    
    # Display distribution of selected prompts by model
    print("\nDistribution by model:")
    model_counts = df_story_gen['model_of_original_prompt'].value_counts()
    for model, count in model_counts.items():
        print(f"{model}: {count} prompts")
    
    # Display distribution by iteration across all models
    print("\nDistribution by iteration (across all models):")
    iteration_counts = df_story_gen['iteration'].value_counts().sort_index()
    for iteration, count in iteration_counts.items():
        print(f"Iteration {iteration}: {count} prompts")
        
    # Display distribution by model and iteration
    print("\nDetailed distribution by model and iteration:")
    model_iter_counts = df_story_gen.groupby(['model_of_original_prompt', 'iteration']).size().unstack(fill_value=0)
    print(model_iter_counts)
    
    # Add a column with the story generation prompt (with prefix)
    df_story_gen["step1_story_prompt_with_prefix"] = [
        f"Please write me a story in exactly 5 sentences: {prompt}"
        for prompt in df_story_gen["prompt"]
    ]
    
    # Calculate prompt length statistics for the selected prompts
    df_story_gen["prompt_length"] = df_story_gen["prompt"].str.len()
    df_story_gen["full_prompt_length"] = df_story_gen["step1_story_prompt_with_prefix"].str.len()
    
    print("\n" + "="*50)
    print("PROMPT LENGTH STATISTICS FOR SELECTED PROMPTS:")
    print("="*50)
    print(f"Original prompts - Avg length: {df_story_gen['prompt_length'].mean():.2f} characters")
    print(f"Original prompts - Min length: {df_story_gen['prompt_length'].min()} characters")
    print(f"Original prompts - Max length: {df_story_gen['prompt_length'].max()} characters")
    print(f"With prefix - Avg length: {df_story_gen['full_prompt_length'].mean():.2f} characters")
    print(f"With prefix - Min length: {df_story_gen['full_prompt_length'].min()} characters")
    print(f"With prefix - Max length: {df_story_gen['full_prompt_length'].max()} characters")
    
    # Display a few example prompts
    print("\n" + "="*50)
    print("SAMPLE OF SELECTED PROMPTS:")
    print("="*50)
    sample_indices = random.sample(range(len(df_story_gen)), min(5, len(df_story_gen)))
    for idx in sample_indices:
        row = df_story_gen.iloc[idx]
        print(f"\nExample {idx+1}:")
        print(f"Model: {row['model_of_original_prompt']}, Iteration: {row['iteration']}, Order: {row.get('order', 'N/A')}")
        print(f"Original prompt: {row['prompt']}")
        print(f"Story generation prompt: {row['step1_story_prompt_with_prefix']}")
        print(f"Length: {row['prompt_length']} chars (original), {row['full_prompt_length']} chars (with prefix)")
    
    # Save the selected prompts for story generation
    output_file = "./data/story_generation/selected_prompts.csv"
    df_story_gen.to_csv(output_file, index=False)
    print(f"\nSelected prompts saved to {output_file}")
    
    return df_story_gen

def main():
    # Create output directory if it doesn't exist
    os.makedirs("./data/story_seeds", exist_ok=True)
    
    # Check if models directory exists
    models_dir = "./data/story_seeds/raw_responses/models"
    if not os.path.exists(models_dir):
        print(f"Error: Directory {models_dir} does not exist!")
        return
    
    # Get all individual model response files
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".json") and not f.endswith("_all_iterations.json")]
    
    if not model_files:
        print("No individual model response files found to process!")
        return
    
    # Process each model file
    all_data = pd.DataFrame()
    
    for file_name in sorted(model_files):
        file_path = os.path.join(models_dir, file_name)
        print(f"Processing {file_path}...")
        
        # Process the file
        df = process_individual_model_file(file_path)
        
        if df.empty:
            print(f"Warning: No data extracted from {file_name}")
            continue
        
        # Display information about the DataFrame
        print(f"\nResults from {file_name}:")
        print(f"Model: {df['model'].iloc[0]}, Iteration: {df['iteration'].iloc[0]}")
        print(f"Total rows: {len(df)}")
        print(f"Number of unique prompts: {df['prompt'].nunique()}")
        
        # Concatenate with the main dataframe
        all_data = pd.concat([all_data, df], ignore_index=True)
    
    # Display information about the combined DataFrame
    if not all_data.empty:
        print("\n" + "="*50)
        print("FINAL COMBINED RESULTS:")
        print("="*50)
        print(f"Total rows: {len(all_data)}")
        print(f"Number of unique prompts: {all_data['prompt'].nunique()}")
        print(f"Number of unique models: {all_data['model'].nunique()}")
        print(f"Number of iterations: {all_data['iteration'].nunique()}")

        # Group by iteration to save individual iteration files
        for iteration, iteration_data in all_data.groupby('iteration'):
            output_file = f"./data/story_seeds/story_seeds_iteration_{iteration}.csv"
            iteration_data.to_csv(output_file, quoting=csv.QUOTE_MINIMAL, index=False, sep='|')
            print(f"Iteration {iteration} data saved to {output_file}")

        # Show counts by model and iteration
        print("\nCounts by model and iteration:")
        model_iteration_counts = all_data.groupby(["model", "iteration"]).size().unstack(fill_value=0)
        print(model_iteration_counts)

        # Calculate and display prompt length statistics
        all_data['prompt_length'] = all_data['prompt'].str.len()
        
        # Display prompt length per model and iteration
        print("\n" + "="*50)
        print("PROMPT LENGTH PER MODEL AND ITERATION:")
        print("="*50)
        model_iteration_lengths = all_data.groupby(["model", "iteration"])['prompt_length'].agg([
            'count', 'mean', 'min', 'max', 'sum'
        ]).reset_index()
        
        # Rename columns for readability
        model_iteration_lengths = model_iteration_lengths.rename(columns={
            'count': 'num_prompts',
            'mean': 'avg_length',
            'min': 'min_length',
            'max': 'max_length',
            'sum': 'total_chars'
        })
        
        # Round the average to 2 decimal places
        model_iteration_lengths['avg_length'] = model_iteration_lengths['avg_length'].round(2)
        
        print(model_iteration_lengths)
        
        # Overall summary of prompt lengths
        print("\n" + "="*50)
        print("OVERALL PROMPT LENGTH SUMMARY:")
        print("="*50)
        print(f"Average prompt length: {all_data['prompt_length'].mean().round(2)} characters")
        print(f"Minimum prompt length: {all_data['prompt_length'].min()} characters")
        print(f"Maximum prompt length: {all_data['prompt_length'].max()} characters")
        print(f"Median prompt length: {all_data['prompt_length'].median().round(2)} characters")
        print(f"Total characters across all prompts: {all_data['prompt_length'].sum()}")
        
        # Get top 5 longest and shortest prompts
        print("\n" + "="*50)
        print("SAMPLE OF LONGEST PROMPTS:")
        print("="*50)
        longest_prompts = all_data.nlargest(5, 'prompt_length')[['model', 'iteration', 'prompt_length', 'prompt']]
        for _, row in longest_prompts.iterrows():
            print(f"Model: {row['model']}, Iteration: {row['iteration']}, Length: {row['prompt_length']}")
            print(f"Prompt: {row['prompt'][:100]}...\n")
            
        print("\n" + "="*50)
        print("SAMPLE OF SHORTEST PROMPTS:")
        print("="*50)
        shortest_prompts = all_data.nsmallest(5, 'prompt_length')[['model', 'iteration', 'prompt_length', 'prompt']]
        for _, row in shortest_prompts.iterrows():
            print(f"Model: {row['model']}, Iteration: {row['iteration']}, Length: {row['prompt_length']}")
            print(f"Prompt: {row['prompt']}\n")
            
        # Save the length statistics
        stats_output_file = "./data/story_seeds/prompt_length_stats.csv"
        model_iteration_lengths.to_csv(stats_output_file, index=False)
        print(f"\nPrompt length statistics saved to {stats_output_file}")

        # Save the combined data
        output_file = "./data/story_seeds/story_seeds_combined.csv"
        all_data.to_csv(output_file, quoting=csv.QUOTE_MINIMAL, index=False, sep='|')
        print(f"\nCombined data saved to {output_file}")
        
        # Now select random prompts for story generation
        print("\n" + "="*50)
        print("SELECTING RANDOM PROMPTS FOR STORY GENERATION:")
        print("="*50)
        df_story_gen = select_random_prompts(all_data)  # Using PER_MODEL_COUNT as default
        output_file = "./data/story_seeds/story_seeds_combined_subsampled.csv"
        df_story_gen.to_csv(output_file, quoting=csv.QUOTE_MINIMAL, index=False, sep='|')

        print("\nPrompt selection complete. Ready for story generation!")
        
    else:
        print("No data was processed!")

if __name__ == "__main__":
    main()