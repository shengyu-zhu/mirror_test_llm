import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from tqdm.asyncio import tqdm as async_tqdm
import random
import os
import asyncio
import sys
from difflib import SequenceMatcher
# Add the parent directory to path so imports work correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# Import api_keys for models
from utils.util import TextProcessor
from utils.util import Grok, Gemini
from utils.util import convert_to_ordinal_text
from utils.util import apply_balanced_random

# Define all prompt templates and paths
DATA_DIR = "./data/"
# Modified to accept variable sentence counts
STORY_PROMPT_TEMPLATE = "please write me a story in exactly {sentence_count} sentences: {prompt}"
REVISE_SENTENCE_PROMPT = "revise only the {ordinal} sentence in your own unique way, answer with only the revised sentence: {text}"
STRANGE_PART_PROMPT = "What is the part of the paragraph you think is strange? Please answer with only the integer order of the sentence from 1-{sentence_count}: {text}"

# Configuration parameters
random_state = 2025
batch_size = 10

# Define sentence counts to test
sentence_counts = [2, 3, 5, 7, 10, 13, 15, 18, 20]

# Enhanced error handling for API calls
async def async_model_call(model_func, prompt, temperature=0.0):
    """Async wrapper for model API calls with error handling"""
    try:
        # Replace the loop.run_in_executor method with direct async call
        # to avoid creating too many threads/connections
        result = await asyncio.to_thread(lambda: model_func(prompt, temperature=temperature))
        
        # Explicitly check if the result is None or empty
        if result is None:
            return (f"ERROR: Received None response", "none_response")
            
        return result
    except Exception as e:
        # Check if the exception has an error code attribute
        error_code = getattr(e, 'code', None) or getattr(e, 'status_code', None) or getattr(e, 'error_code', None)
        
        # Helpful error name for debugging
        error_name = type(e).__name__
        
        # Enhanced error reporting
        if error_code:
            print(f"\nAPI ERROR ({error_name}): {error_code} - {str(e)}", file=sys.stderr)
        else:
            print(f"\nAPI ERROR ({error_name}): {str(e)}", file=sys.stderr)
            
        # Check specifically for connection errors
        if "ConnectionError" in error_name or "too many open files" in str(e).lower():
            print("Connection error detected - pausing to allow connections to close...")
            await asyncio.sleep(5)  # Add extra pause for connection errors
            
        # Return a tuple with error information that can be detected later
        return (f"ERROR: {str(e)}", error_code or error_name)

async def process_batch_with_model(model_func, prompts, temperature=0.0, batch_size=5, description="Processing", max_retries=5, retry_delay=3):
    """Process a batch of prompts concurrently with throttling, progress tracking, and error handling"""
    results = []
    # Add progress bar for overall batch processing
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    with async_tqdm(total=total_batches, desc=description) as pbar:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            retry_count = 0
            batch_results = None
            
            while retry_count < max_retries:
                tasks = [async_model_call(model_func, prompt, temperature) for prompt in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check if there are rate limit errors that we should retry
                rate_limit_errors = [
                    idx for idx, result in enumerate(batch_results) 
                    if isinstance(result, tuple) and 
                    isinstance(result[0], str) and 
                    result[0].startswith("ERROR") and
                    any(code in str(result) for code in ["429", "rate_limit", "too_many_requests"])
                ]
                
                if rate_limit_errors:
                    retry_count += 1
                    batch_size_new = max(1, batch_size // 2)  # Reduce batch size for retries
                    print(f"\nRate limit hit, retrying with smaller batch size ({batch_size_new}). Retry {retry_count}/{max_retries}")
                    await asyncio.sleep(retry_delay * retry_count)  # Exponential backoff
                    # Only retry the failed requests
                    batch = [batch[idx] for idx in rate_limit_errors]
                    # Keep successful results
                    results.extend([r for idx, r in enumerate(batch_results) if idx not in rate_limit_errors])
                    batch_size = batch_size_new
                else:
                    # No rate limit errors, add all results and break retry loop
                    results.extend(batch_results)
                    break
            
            # Add a larger delay between batches to avoid rate limiting and connection issues
            if i + batch_size < len(prompts):
                # Dynamic delay based on batch size to prevent overwhelming API
                delay = min(2.0, 0.2 * batch_size)
                await asyncio.sleep(delay)
            
            # Call garbage collection explicitly after each batch
            import gc
            gc.collect()
            
            pbar.update(1)
    
    # Check if any errors are in the results and display them
    error_count = 0
    for idx, result in enumerate(results):
        if isinstance(result, tuple) and isinstance(result[0], str) and result[0].startswith("ERROR"):
            error_count += 1
            print(f"Error in result {idx}: {result}", file=sys.stderr)
    
    if error_count > 0:
        print(f"\nCompleted with {error_count} errors out of {len(results)} API calls", file=sys.stderr)
    
    return results

async def run_experiment_for_sentence_count(sentence_count, df_story_gen, m1, m2, m1_temperature, m2_temperature):
    """Run the experiment for a specific sentence count with specified models and temperatures"""
    print(f"\n{'='*80}\nRunning experiment with {sentence_count} sentences\n{'='*80}")
    
    model1 = m1.__name__
    model2 = m2.__name__
    text_processor = TextProcessor()
    
    # Create output directory if it doesn't exist
    output_dir = f"{DATA_DIR}sentence_count_result"
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = f"{output_dir}/mirror_test_results_{model1}_{model2}_{sentence_count}_sentences.csv"
    if os.path.exists(file_path):
        print(f"File {file_path} already exists. Skipping this sentence count.")
        # Optionally load and return the existing results
        try:
            existing_df = pd.read_csv(file_path, sep='|')
            if 'is_correct' in existing_df.columns:
                accuracy = existing_df['is_correct'].mean()
                return {
                    "sentence_count": sentence_count,
                    "model1": model1,
                    "model2": model2,
                    "accuracy": accuracy,
                    "sample_size": len(existing_df),
                    "temperature1": m1_temperature,
                    "temperature2": m2_temperature
                }
        except Exception as e:
            print(f"Error loading existing results: {e}")
        return None
    
    # Update the prompt template for the current sentence count
    df_m1 = df_story_gen.copy()  # Use full dataset
    df_m1.loc[:, "step1_story_prompt_with_prefix"] = [
        STORY_PROMPT_TEMPLATE.format(sentence_count=sentence_count, prompt=prompt)
        for prompt in df_m1["prompt"]
    ]
        
    # Step 1: Generate stories
    print(f"Generate initial stories with {sentence_count} sentences using {model1}")
    
    prompts = df_m1["step1_story_prompt_with_prefix"].tolist()
    print(f"Processing {len(prompts)} prompts in batches of {batch_size}")
    outputs = await process_batch_with_model(
        m1, 
        prompts, 
        m1_temperature, 
        batch_size,
        f"Generating {sentence_count}-sentence stories with {model1}"
    )
    
    # Process outputs
    df_m1.loc[:, "step1_m1_output"] = outputs
    df_m1.loc[:, "step1_m1_output_sentence_only"] = [
        text[0] if not (isinstance(text, tuple) and isinstance(text[0], str) and text[0].startswith("ERROR")) else None 
        for text in df_m1['step1_m1_output']
    ]
    df_m1.loc[:, "step1_m1_output_sent_count"] = df_m1["step1_m1_output_sentence_only"].apply(
        lambda x: text_processor.count_sentences(x) if x is not None else None
    )
    df_m1.loc[:, "requested_sentence_count"] = sentence_count
    
    # Save intermediate results
    step1_file = f"{output_dir}/step1_generate_stories_{model1}_{sentence_count}_sentences_{m1_temperature}temp.csv"
    df_m1.to_csv(step1_file, index=False, sep='|')
    print(f"Step 1 results saved to {step1_file}")
    
    # Report on sentence count accuracy
    stories_with_correct_count = df_m1[df_m1["step1_m1_output_sent_count"] == sentence_count]
    print(f"Model {model1} generated {len(stories_with_correct_count)} stories with exactly {sentence_count} sentences " 
          f"({len(stories_with_correct_count) / len(df_m1) * 100:.1f}% accuracy)")
    
    # Use stories_with_correct_count instead of df_m1 for further processing
    # Add step2_random_sent_num
    stories_with_correct_count = apply_balanced_random(
        stories_with_correct_count, 
        count_column="step1_m1_output_sent_count",
        output_column="step2_random_sent_num", 
        random_state=random_state
    )
    stories_with_correct_count.loc[:, "step2_random_sent_num_ordinal_text"] = convert_to_ordinal_text(stories_with_correct_count, 'step2_random_sent_num')
    stories_with_correct_count.loc[:, "step2_marked_text_input_nth_sentence"] = stories_with_correct_count.apply(
        lambda row: REVISE_SENTENCE_PROMPT.format(
            ordinal=row['step2_random_sent_num_ordinal_text'], 
            text=row['step1_m1_output_sentence_only']
        ) if row['step2_random_sent_num_ordinal_text'] is not None else None,
        axis=1)

    # Calculate average text length
    valid_texts = [x for x in stories_with_correct_count['step1_m1_output_sentence_only'].tolist() if isinstance(x, str)]
    if valid_texts:
        avg_length = np.average([len(x) for x in valid_texts])
        print(f"Average length of stories: {avg_length:.1f} characters")
    else:
        print("No valid texts found to calculate average length")

    # Now we already have stories with the correct sentence count, so we don't need to filter again
    valid_sentence_count_df = stories_with_correct_count.dropna(subset=['step2_marked_text_input_nth_sentence'])

    # Use all valid stories with the correct sentence count
    print(f"Using all {len(valid_sentence_count_df)} stories with exactly {sentence_count} sentences.")

    # No sampling - use all available data
    sampled_df = valid_sentence_count_df

    if len(sampled_df) == 0:
        print(f"No valid samples found for {sentence_count} sentences. Skipping.")
        return {
            "sentence_count": sentence_count,
            "model1": model1,
            "model2": model2,
            "accuracy": None,
            "sample_size": 0,
            "temperature1": m1_temperature,
            "temperature2": m2_temperature,
            "error": "No valid samples"
        }
    
    # Step 2: Get second model's revisions
    print(f"Step 2: Getting {model2} to revise sentences in {sentence_count}-sentence stories")
    prompts = sampled_df["step2_marked_text_input_nth_sentence"].dropna().tolist()
    
    if prompts:
        outputs = await process_batch_with_model(
            m2, 
            prompts, 
            m2_temperature, 
            batch_size,
            f"Revising sentences with {model2}"
        )
        
        # Create mapping from prompt to output
        prompt_to_output = {p: o for p, o in zip(prompts, outputs)}
        
        # Assign outputs back to dataframe
        sampled_df.loc[:, "step2_output_nth_sentence"] = sampled_df["step2_marked_text_input_nth_sentence"].map(
            lambda x: prompt_to_output.get(x, None)
        )
        
        # Handle potential errors in outputs
        sampled_df.loc[:, "step2_output_nth_sentence_message_only"] = sampled_df["step2_output_nth_sentence"].apply(
            lambda x: x[0] if x is not None and not (
                isinstance(x, tuple) and isinstance(x[0], str) and x[0].startswith("ERROR")
            ) else None
        )
        
        # Replace sentences in original stories
        sampled_df.loc[:, "step2_output"] = sampled_df.apply(
            lambda row: text_processor.replace_nth_sentence(
                row["step1_m1_output_sentence_only"],
                row["step2_random_sent_num"],
                row["step2_output_nth_sentence_message_only"]
            ) if all(pd.notna([
                row["step1_m1_output_sentence_only"], 
                row["step2_random_sent_num"], 
                row["step2_output_nth_sentence_message_only"]
            ])) else None,
            axis=1
        )
        
        # Filter valid rows for step 3
        valid_df = sampled_df.dropna(subset=['step2_output'])
        
        if len(valid_df) == 0:
            print(f"No valid samples after sentence replacement for {sentence_count} sentences. Skipping.")
            return {
                "sentence_count": sentence_count,
                "model1": model1,
                "model2": model2,
                "accuracy": None,
                "sample_size": 0,
                "temperature1": m1_temperature,
                "temperature2": m2_temperature,
                "error": "No valid samples after sentence replacement"
            }
            
        # Step 3: Update the strange part prompt to include the correct sentence count
        valid_df.loc[:, "step3_input"] = [
            STRANGE_PART_PROMPT.format(sentence_count=sentence_count, text=text)
            for text in valid_df["step2_output"]
        ]
        
        # Find strange sentences
        print(f"Step 3: Asking {model1} to find modified sentences in {sentence_count}-sentence stories")
        prompts = valid_df["step3_input"].tolist()
        outputs = await process_batch_with_model(
            m1, 
            prompts, 
            m1_temperature, 
            batch_size,
            f"Finding strange sentences"
        )
        
        # Create mapping from input to output
        input_to_output = {i: o for i, o in zip(prompts, outputs)}
        
        # Assign outputs back to dataframe
        valid_df.loc[:, "step3_output"] = valid_df["step3_input"].map(
            lambda x: input_to_output.get(x, None)
        )
        
        valid_df.loc[:, "step3_output_message_only"] = valid_df["step3_output"].apply(
            lambda x: x[0] if x is not None and not (
                isinstance(x, tuple) and isinstance(x[0], str) and x[0].startswith("ERROR")
            ) else None
        )

        # Calculate accuracy using numeric conversion with error handling
        try:
            # Filter to rows with valid output
            valid_for_accuracy = valid_df.dropna(subset=['step3_output_message_only', 'step2_random_sent_num'])
            
            if len(valid_for_accuracy) > 0:
                # Convert to int with error handling
                valid_for_accuracy.loc[:, 'step3_output_int'] = pd.to_numeric(
                    valid_for_accuracy["step3_output_message_only"], 
                    errors='coerce'
                ).fillna(0).astype(int)
                
                # Calculate accuracy directly
                accuracy = sum(valid_for_accuracy['step3_output_int'] == valid_for_accuracy["step2_random_sent_num"])/len(valid_for_accuracy)
                print(f"Accuracy for {sentence_count} sentences: {accuracy:.4f} ({accuracy*100:.2f}%)")
                
                # Add is_correct column to the dataframe for later analysis
                valid_df.loc[:, 'step3_output_int'] = pd.to_numeric(
                    valid_df["step3_output_message_only"], 
                    errors='coerce'
                ).fillna(0).astype(int)
                
                valid_df.loc[:, 'is_correct'] = valid_df['step3_output_int'] == valid_df['step2_random_sent_num']
                valid_df.loc[:, 'model1'] = model1
                valid_df.loc[:, 'model2'] = model2
                valid_df.loc[:, 'temp1'] = m1_temperature
                valid_df.loc[:, 'temp2'] = m2_temperature

                # Save final results
                output_file = f"{output_dir}/mirror_test_results_{model1}_{model2}_{sentence_count}_sentences.csv"
                valid_df.to_csv(output_file, index=False, sep='|')
                print(f"Results saved to {output_file}")
                
                return {
                    "sentence_count": sentence_count,
                    "model1": model1,
                    "model2": model2,
                    "accuracy": accuracy,
                    "sample_size": len(valid_for_accuracy),
                    "temperature1": m1_temperature,
                    "temperature2": m2_temperature
                }
            else:
                print("No valid data for accuracy calculation")
                return {
                    "sentence_count": sentence_count,
                    "model1": model1,
                    "model2": model2,
                    "accuracy": None,
                    "sample_size": 0,
                    "temperature1": m1_temperature,
                    "temperature2": m2_temperature,
                    "error": "No valid data for accuracy calculation"
                }
        except Exception as e:
            print(f"Error calculating accuracy: {str(e)}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return {
                "sentence_count": sentence_count,
                "model1": model1,
                "model2": model2,
                "accuracy": None,
                "sample_size": 0,
                "temperature1": m1_temperature,
                "temperature2": m2_temperature,
                "error": str(e)
            }
    else:
        print(f"No valid prompts for {sentence_count} sentences. Skipping.")
        return {
            "sentence_count": sentence_count,
            "model1": model1,
            "model2": model2,
            "accuracy": None,
            "sample_size": 0,
            "temperature1": m1_temperature,
            "temperature2": m2_temperature,
            "error": "No valid prompts for sentence modification"
        }

# Import the generate_dataset function
def load_dataset():
    """
    Load the dataset from the default location
    
    Returns:
        pandas.DataFrame: The loaded dataset
    """
    print("\n" + "="*80)
    print(f"LOADING EXISTING DATASET")
    print("="*80)
    
    # Default file path
    dataset_path = "./data/mirror_test/mirror_test_dataset.csv"
    
    try:
        # Try loading with pipe separator first
        df = pd.read_csv(dataset_path, sep='|')
    except Exception as e:
        print(f"Error loading with pipe separator: {e}")
        # Try with auto-detection as fallback
        try:
            df = pd.read_csv(dataset_path, sep=None, engine='python')
        except Exception as e2:
            print(f"Error loading dataset: {e2}")
            print("Falling back to original dataset source...")
            # Try loading from the original path as last resort
            original_path = "./data/story_seeds/story_seeds_combined.csv"
            try:
                df = pd.read_csv(original_path, sep='|')
            except:
                df = pd.read_csv(original_path, sep=None, engine='python')
    
    print(f"Dataset loaded with {len(df)} samples")
    
    # Ensure the required columns exist
    if 'model' in df.columns and 'model_of_original_prompt' not in df.columns:
        df = df.rename(columns={'model': 'model_of_original_prompt'})
    
    return df

async def run_sentence_count_experiments(models_to_test):
    """Run experiments across different sentence counts with specified model combinations"""
    print("Loading data...")
    try:
        # Create output directory
        os.makedirs(f"{DATA_DIR}sentence_count_result", exist_ok=True)
        
        # Load the dataset instead of generating it
        df_story_gen = load_dataset()
        
        # Check if we have data to work with
        if df_story_gen is None or len(df_story_gen) == 0:
            print("Failed to load a valid dataset. Exiting.")
            return
            
        print(f"Using dataset with {len(df_story_gen)} samples")
        
        # Rename model column if it exists
        if 'model' in df_story_gen.columns:
            df_story_gen = df_story_gen.rename(columns={'model': 'model_of_original_prompt'})
        
        # No duplicate copies - using the dataset as is
        print(f"Using dataset with {len(df_story_gen)} samples")
        
        results = []
        
        # For each model combination
        for model_config in models_to_test:
            m1 = model_config["model1"]
            m2 = model_config["model2"]
            temp1 = model_config["temp1"]
            temp2 = model_config["temp2"]
            
            print(f"\n{'#'*80}\nTesting {m1.__name__} (temp={temp1}) vs {m2.__name__} (temp={temp2})\n{'#'*80}")
            
            # Run for each sentence count sequentially to avoid overwhelming the API
            for count in sentence_counts:
                result = await run_experiment_for_sentence_count(
                    count, df_story_gen, m1, m2, temp1, temp2
                )
                if result:
                    results.append(result)
                    
                    # Save intermediate summary after each experiment
                    if results:
                        summary_df = pd.DataFrame(results)
                        summary_file = f"{DATA_DIR}sentence_count_result/accuracy_summary_by_sentence_count.csv"
                        summary_df.to_csv(summary_file, index=False)
                        print(f"Updated summary saved to {summary_file}")
                
                # Free memory explicitly and pause between runs
                import gc
                gc.collect()
                
                # Longer pause between sentence count tests
                print(f"Pausing between sentence count tests to allow connections to close...")
                await asyncio.sleep(5)  # Increased pause between sentence count tests
        
        # Compile final results across all sentence counts
        if results:
            print("\nFinal results across all experiments:")
            summary_df = pd.DataFrame(results)
            
            # Create a pivot table to better visualize results
            if len(summary_df) > 1:
                try:
                    pivot_df = summary_df.pivot_table(
                        index=['model1', 'model2', 'temperature1', 'temperature2'],
                        columns='sentence_count',
                        values='accuracy',
                        aggfunc='mean'
                    )
                    pivot_file = f"{DATA_DIR}sentence_count_result/accuracy_pivot_by_sentence_count.csv"
                    pivot_df.to_csv(pivot_file)
                    print(f"Pivot table saved to {pivot_file}")
                    print("\nAccuracy by sentence count:")
                    print(pivot_df)
                except Exception as e:
                    print(f"Error creating pivot table: {e}")
            
            print("\nDetailed results:")
            print(summary_df)
            
            summary_file = f"{DATA_DIR}sentence_count_result/accuracy_summary_by_sentence_count.csv"
            summary_df.to_csv(summary_file, index=False)
            print(f"Final summary saved to {summary_file}")
        else:
            print("No valid results to compile.")
    
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

async def main():
    # Define model combinations to test
    models_to_test = [
        {"model1": Grok, "model2": Gemini, "temp1": 0.7, "temp2": 0.7},
    ]
    
    # Run the experiments using the pre-generated dataset
    await run_sentence_count_experiments(models_to_test)
    
    # Print summary of what was done
    print("\n" + "="*80)
    print("MIRROR TEST EXPERIMENT COMPLETED")
    print("="*80)
    print("Summary of operations:")
    print(f"1. Used pre-generated dataset from ./data/mirror_test/mirror_test_dataset.csv")
    print("2. Ran mirror test experiments across sentence counts:", sentence_counts)
    print("3. Model combinations tested:", ", ".join([f"{m['model1'].__name__} vs {m['model2'].__name__}" for m in models_to_test]))
    print(f"4. Results saved to {DATA_DIR}sentence_count_result/")
    print("="*80)

# Set a higher limit on open files before running
def increase_file_limit():
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"Current file limits: soft={soft}, hard={hard}")
        
        # Try to increase to hard limit
        if soft < hard:
            # Set to a value that's less than hard but higher than default
            # This is more likely to succeed than trying to hit the hard limit exactly
            new_limit = min(4096, hard - 100)
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, hard))
            new_soft, new_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            print(f"New file limits: soft={new_soft}, hard={new_hard}")
        else:
            print("Already at maximum file limit")
            
        # Verify the change took effect
        import subprocess
        try:
            ulimit_output = subprocess.check_output("ulimit -n", shell=True, text=True)
            print(f"System reports ulimit -n: {ulimit_output.strip()}")
        except:
            pass
    except Exception as e:
        print(f"Failed to increase file limit: {e}")
        
    # Also set a suggested limit for concurrent connections in httpx if available
    try:
        import httpx
        httpx.Limits(max_connections=100, max_keepalive_connections=20)
        print("Set httpx connection limits")
    except Exception as e:
        print(f"Could not set httpx limits: {e}")

if __name__ == "__main__":
    # Increase file limit first
    increase_file_limit()
    
    # Run the async main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"FATAL ERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)