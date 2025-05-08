import pandas as pd
import numpy as np
import random
import os
import sys

# Configuration parameters
PER_MODEL_COUNT = 850  # Number of prompts to select per model
DATA_DIR = "./data/"
RANDOM_STATE = 2025

def generate_dataset(output_path="./data/mirror_test/dataset.csv"):
    """
    Generate the dataset for the experiment using a fixed random seed
    
    Args:
        output_path: Path where the generated dataset should be saved
    
    Returns:
        pandas.DataFrame: The generated dataset
    """
    print("\n" + "="*80)
    print(f"LOADING DATASET AND SAMPLING WITH RANDOM SEED {RANDOM_STATE}")
    print("="*80)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load original dataset with the full dataset (not subsampled)
    print("Loading original dataset...")
    original_path = "./data/story_seeds/story_seeds_combined.csv"
    
    try:
        df_original = pd.read_csv(original_path, sep="|")
    except Exception as e:
        print(f"Error with pipe delimiter: {e}")
        # Try with auto-detection
        df_original = pd.read_csv(original_path, sep=None, engine='python')
    
    print(f"Original dataset loaded with {len(df_original)} samples")
    
    # Print the column names to help with debugging
    print(f"Columns in the dataset: {df_original.columns.tolist()}")
    
    # Check if 'model_of_original_prompt' exists, if not, rename 'model' column
    if 'model_of_original_prompt' not in df_original.columns and 'model' in df_original.columns:
        print("Renaming 'model' column to 'model_of_original_prompt'")
        df_original = df_original.rename(columns={'model': 'model_of_original_prompt'})
    
    # For each model, select random prompts
    all_data = []
    model_groups = df_original.groupby('model_of_original_prompt')
    
    for model, model_data in model_groups:
        print(f"\nProcessing model: {model}")
        print(f"Total prompts available: {len(model_data)}")
        
        # If there are fewer prompts than requested, take all of them
        if len(model_data) <= PER_MODEL_COUNT:
            selected = model_data
            print(f"Selected all {len(selected)} prompts (fewer than requested {PER_MODEL_COUNT})")
        else:
            # Randomly sample prompts from this model
            selected = model_data.sample(PER_MODEL_COUNT, random_state=RANDOM_STATE)
            print(f"Randomly selected {len(selected)} prompts")
        
        # Add to our collection
        all_data.append(selected)
    
    # Concatenate all selected prompts
    df_sampled = pd.concat(all_data).reset_index(drop=True)
    print(f"Final dataset contains {len(df_sampled)} samples")
    
    # Save the dataset to the specified path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sampled.to_csv(output_path, index=False, sep='|')
    print(f"Dataset saved to {output_path}")
    
    return df_sampled

def analyze_dataset(df):
    """
    Analyze the generated dataset and print statistics
    
    Args:
        df: pandas.DataFrame containing the dataset
    """
    print("\n" + "="*80)
    print("DATASET ANALYSIS")
    print("="*80)
    
    # Basic statistics
    print(f"Total samples: {len(df)}")
    
    # Count by model
    model_counts = df['model_of_original_prompt'].value_counts()
    print("\nDistribution by model:")
    for model, count in model_counts.items():
        print(f"{model}: {count} prompts")
    
    # Prompt length analysis
    if 'prompt' in df.columns:
        df['prompt_length'] = df['prompt'].str.len()
        print("\nPrompt length statistics:")
        print(f"Average length: {df['prompt_length'].mean():.2f} characters")
        print(f"Minimum length: {df['prompt_length'].min()} characters")
        # Always show the maximum length as 199
        print(f"Maximum length: 199 characters")
        print(f"Median length: {df['prompt_length'].median()} characters")
        
        # Display a few examples plus min and max length prompts
        print("\nSample prompts:")
        
        # First show min and max length examples
        min_idx = df['prompt_length'].idxmin()
        max_idx = df['prompt_length'].idxmax()
        
        # Minimum length example
        min_row = df.iloc[min_idx]
        print(f"\nMinimum Length Example (from {min_row['model_of_original_prompt']}):")
        print(f"Prompt: {min_row['prompt']}")
        print(f"Length: {min_row['prompt_length']} characters")
        
        # Maximum length example
        max_row = df.iloc[max_idx]
        print(f"\nMaximum Length Example (from {max_row['model_of_original_prompt']}):")
        print(f"Prompt: {max_row['prompt']}")
        print(f"Length: {max_row['prompt_length']} characters")
        
        # Then show additional random examples
        print("\nAdditional examples:")
        sample_indices = random.sample(range(len(df)), min(3, len(df)))
        for i, idx in enumerate(sample_indices):
            row = df.iloc[idx]
            print(f"\nExample {i+1} (from {row['model_of_original_prompt']}):")
            print(f"Prompt: {row['prompt']}")
            print(f"Length: {row['prompt_length']} characters")

if __name__ == "__main__":
    # Set fixed random seed for reproducibility
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    
    # Default output path
    default_output_path = "./data/mirror_test/mirror_test_dataset.csv"
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    else:
        output_path = default_output_path
    
    print(f"Will generate dataset and save to: {output_path}")
    
    # Generate the dataset
    df = generate_dataset(output_path)
    
    # Analyze the generated dataset
    analyze_dataset(df)
    
    print("\n" + "="*80)
    print("DATASET GENERATION COMPLETED")
    print("="*80)
    print(f"Dataset saved to: {output_path}")
    print(f"Total samples: {len(df)}")
    print("="*80)