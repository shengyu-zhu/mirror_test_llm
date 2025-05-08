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
from utils.util import TextProcessor
from utils.util import Gemini, Claude, Grok, ChatGPT
from utils.util import convert_to_ordinal_text, apply_balanced_random

# Define all prompt templates and paths at the top of the script
DATA_DIR = "./data/"
STORY_PROMPT_PREFIX = "please write me a story in exactly 5 setences: "
REVISE_SENTENCE_PROMPT = "revise only the {ordinal} sentence in your own unique way, answer with only the revised sentence: {text}"
STRANGE_PART_PROMPT = "What is the part of the paragraph you think is strange? Please answer with only the integer order of the sentence from 1-5: {text}"

n_samples = 1000
random_state = 2025
batch_size = 10  # Adjust based on API rate limits

# Enhanced error handling for API calls
async def async_model_call(model_func, prompt, temperature=0.0):
    """Async wrapper for model API calls with error handling"""
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None, lambda: model_func(prompt, temperature=temperature)
        )
        return result
    except Exception as e:
        # Check if the exception has an error code attribute
        error_code = getattr(e, 'code', None) or getattr(e, 'status_code', None) or getattr(e, 'error_code', None)
        if error_code:
            print(f"\nAPI ERROR: {error_code} - {str(e)}", file=sys.stderr)
        else:
            print(f"\nAPI ERROR: {str(e)}", file=sys.stderr)
        # Return a tuple with error information that can be detected later
        return (f"ERROR: {str(e)}", error_code)

async def process_batch_with_model(model_func, prompts, temperature=0.0, batch_size=10, description="Processing", max_retries=3, retry_delay=2):
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
            
            # Add a small delay between batches to avoid rate limiting
            if i + batch_size < len(prompts):
                await asyncio.sleep(0.5)
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

async def main():
    print("Loading data...")
    try:
        df = pd.read_csv(f"{DATA_DIR}story_seeds/story_seeds_combined_subsampled.csv", sep="|")
        df = df.rename(columns={'model': 'model_of_original_prompt'})
        df_story_gen = df.copy()
        df_story_gen.loc[:, "step1_story_prompt_with_prefix"] = [
            f"{STORY_PROMPT_PREFIX}{prompt}"
            for prompt in df_story_gen["prompt"]
        ]
        text_processor = TextProcessor()

        # Progress tracking for parameter exploration
        parameter_space = np.round(np.linspace(0, 1, 3).tolist(), 2)
        total_combinations = len(parameter_space) * len(parameter_space)
        current_progress = 0
        print(f"Starting experiment with {total_combinations} temperature combinations")
        
        for idx, m1_temperature_value in enumerate(parameter_space):
            for jdx, m2_temperature_value in enumerate(parameter_space):
                # Display overall progress
                current_progress += 1  # Increment counter
                print(f"Progress: {current_progress}/{total_combinations} combinations ({current_progress/total_combinations*100:.1f}%)")
                
                # m1 = Gemini
                # m2 = Claude
                m1 = Grok
                m2 = Gemini
                # m1 = Grok
                # m2 = ChatGPT
                print(f"Generate result for m1_temperature_value {m1_temperature_value}, m2_temperature_value {m2_temperature_value}")
                model1 = m1.__name__
                model2 = m2.__name__
                file_path = f"{DATA_DIR}temperature_result/mirror_test_results_{model1}_{model2}_{m1_temperature_value}_{m2_temperature_value}.csv"
                
                if os.path.exists(file_path):
                    print(f"File {file_path} already exists. Skipping the loop.")
                    continue
                    
                # Generate Stories
                print(f"Generate initial stories for {model1}")
                # df_m1 = df_story_gen.sample(100, random_state=random_state).copy()
                df_m1 = df_story_gen.copy()
                
                # Async batch processing for initial story generation
                prompts = df_m1["step1_story_prompt_with_prefix"].tolist()
                print(f"Processing {len(prompts)} prompts in batches of {batch_size}")
                outputs = await process_batch_with_model(
                    m1, 
                    prompts, 
                    m1_temperature_value, 
                    batch_size,
                    f"Generating stories with {model1} (temp={m1_temperature_value})"
                )
                
                # Check if there were API errors in the outputs
                has_errors = any(isinstance(output, tuple) and isinstance(output[0], str) and output[0].startswith("ERROR") for output in outputs)
                if has_errors:
                    error_count = sum(1 for output in outputs if isinstance(output, tuple) and isinstance(output[0], str) and output[0].startswith("ERROR"))
                    print(f"WARNING: {error_count} API errors detected in outputs. Check logs above for details.", file=sys.stderr)
                    # Replace error responses with None to continue processing
                    outputs = [None if (isinstance(output, tuple) and isinstance(output[0], str) and output[0].startswith("ERROR")) else output for output in outputs]
                
                df_m1.loc[:, "step1_m1_output"] = outputs
                # Handle None values in output
                df_m1.loc[:, "step1_m1_output_sentence_only"] = [text[0] for text in df_m1['step1_m1_output']]
                df_m1.loc[:, "step1_m1_output_sent_count"] = df_m1["step1_m1_output_sentence_only"].apply(text_processor.count_sentences)
                
                print(f"Saving step 1 results to CSV...")
                df_m1.to_csv(f"{DATA_DIR}temperature_result/step1_generate_stories_{model1}.csv", index=False)

                # Add random part for step 2
                print("Preparing step 2 data...")
                df = pd.read_csv(f"{DATA_DIR}temperature_result/step1_generate_stories_{model1}.csv")
                df = apply_balanced_random(
                    df, 
                    count_column="step1_m1_output_sent_count",
                    output_column="step2_random_sent_num", 
                    random_state=random_state
                )
                df.loc[:, "step2_random_sent_num_ordinal_text"] = convert_to_ordinal_text(df, 'step2_random_sent_num')
                df.loc[:, "step2_marked_text_input_nth_sentence"] = df.apply(
                    lambda row: REVISE_SENTENCE_PROMPT.format(
                        ordinal=row['step2_random_sent_num_ordinal_text'], 
                        text=row['step1_m1_output_sentence_only']
                    ) if row['step2_random_sent_num_ordinal_text'] is not None else None,
                    axis=1
                )
                df.to_csv(f"{DATA_DIR}temperature_result/mirror_test_results_add_random_result_{model1}.csv", index=False)
                print(f"Average length of the step1_m1_output_sentence_only: {np.average([len(x) for x in df['step1_m1_output_sentence_only'].to_list() if isinstance(x, str)])}")

                # Get the result of subset with only 5 stories
                print("Sampling 5-sentence stories...")
                df = pd.read_csv(f"{DATA_DIR}temperature_result/mirror_test_results_add_random_result_{model1}.csv")
                random.seed(2025)
                # Filter out rows with None values in critical columns
                valid_df = df.dropna(subset=['step2_random_sent_num', 'step2_marked_text_input_nth_sentence'])
                try:
                    sampled_df = valid_df.groupby("model_of_original_prompt").apply(
                        lambda x: x.sample(min(n_samples, len(x)), random_state=random_state)
                    ).reset_index(drop=True)
                except ValueError as e:
                    print(f"Error in sampling: {e}", file=sys.stderr)
                    # If there's not enough data for sampling, use what we have
                    sampled_df = valid_df
                sampled_df.to_csv(f"{DATA_DIR}temperature_result/mirror_test_results_add_random_result_exactly_5_sentences_{model1}.csv", index=False)

                # Step 2: Sentence Revision
                print(f"Step 2: {model1} vs {model2}")
                df = pd.read_csv(f"{DATA_DIR}temperature_result/mirror_test_results_add_random_result_exactly_5_sentences_{model1}.csv")
                
                # Async batch processing for sentence revision
                prompts = df["step2_marked_text_input_nth_sentence"].dropna().tolist()
                print(f"Processing {len(prompts)} sentence revisions in batches of {batch_size}")
                outputs = await process_batch_with_model(
                    m2, 
                    prompts, 
                    m2_temperature_value, 
                    batch_size,
                    f"Revising sentences with {model2} (temp={m2_temperature_value})"
                )
                
                # Create a mapping from prompt to output
                prompt_to_output = {p: o for p, o in zip(prompts, outputs)}
                
                # Assign outputs back to dataframe, handling any rows that might have been dropped
                df.loc[:, "step2_output_nth_sentence"] = df["step2_marked_text_input_nth_sentence"].map(
                    lambda x: prompt_to_output.get(x, None)
                )
                
                # Handle potential errors in outputs
                df.loc[:, "step2_output_nth_sentence_messsage_only"] = df["step2_output_nth_sentence"].apply(
                    lambda x: x[0] if x is not None and not (isinstance(x, tuple) and isinstance(x[0], str) and x[0].startswith("ERROR")) else None
                )
                
                print("Replacing sentences in original stories...")
                # Only process rows with valid data
                df.loc[:, "step2_output"] = df.apply(
                    lambda row: text_processor.replace_nth_sentence(
                        row["step1_m1_output_sentence_only"],
                        row["step2_random_sent_num"],
                        row["step2_output_nth_sentence_messsage_only"]
                    ) if all(pd.notna([
                        row["step1_m1_output_sentence_only"], 
                        row["step2_random_sent_num"], 
                        row["step2_output_nth_sentence_messsage_only"]
                    ])) else None,
                    axis=1
                )
                
                # Filter out rows with None in step2_output
                valid_df = df.dropna(subset=['step2_output'])
                
                # Add step3_input to all rows in df first, but with NaN for invalid rows
                df.loc[:, "step3_input"] = None
                # Now set the values for valid rows
                for idx, row in valid_df.iterrows():
                    df.loc[idx, "step3_input"] = STRANGE_PART_PROMPT.format(text=row["step2_output"])
                
                # Step 3: Finding strange sentences
                print("Step 3: Identifying modified sentences...")
                # Get only the non-null prompts
                prompts = df["step3_input"].dropna().tolist()
                
                if prompts:  # Only proceed if there are valid prompts
                    outputs = await process_batch_with_model(
                        m1, 
                        prompts, 
                        m1_temperature_value, 
                        batch_size,
                        f"Finding strange sentences with {model1} (temp={m1_temperature_value})"
                    )
                    
                    # Create a mapping from step3_input to output
                    input_to_output = {i: o for i, o in zip(prompts, outputs)}
                    
                    # Initialize step3_output column with None
                    df.loc[:, "step3_output"] = None
                    
                    # Only map outputs for rows that have a valid step3_input
                    mask = df["step3_input"].notna()
                    df.loc[mask, "step3_output"] = df.loc[mask, "step3_input"].map(
                        lambda x: input_to_output.get(x, None)
                    )
                    
                    # Handle potential errors and extract message
                    df.loc[:, "step3_output_message_only"] = df["step3_output"].apply(
                        lambda x: x[0] if x is not None and not (isinstance(x, tuple) and isinstance(x[0], str) and x[0].startswith("ERROR")) else None
                    )
                    
                    # Calculate accuracy only on valid rows
                    valid_for_accuracy = df.dropna(subset=['step3_output_message_only', 'step2_random_sent_num'])
                    
                    if len(valid_for_accuracy) > 0:
                        try:
                            # Convert to int with error handling
                            valid_for_accuracy['step3_output_int'] = pd.to_numeric(
                                valid_for_accuracy["step3_output_message_only"], 
                                errors='coerce'
                            ).fillna(0).astype(int)
                            
                            accuracy = sum(valid_for_accuracy['step3_output_int'] == valid_for_accuracy["step2_random_sent_num"])/len(valid_for_accuracy)
                            print(f"Final accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                        except Exception as e:
                            print(f"Error calculating accuracy: {str(e)}", file=sys.stderr)
                            accuracy = 0
                    else:
                        print("No valid data for accuracy calculation", file=sys.stderr)
                        accuracy = 0
                else:
                    print("No valid prompts for step 3. Skipping.", file=sys.stderr)
                    accuracy = 0
                
                output_file = f"{DATA_DIR}temperature_result/mirror_test_results_{model1}_{model2}_{m1_temperature_value}_{m2_temperature_value}.csv"
                df.to_csv(output_file, index=False)
                print(f"Pipeline complete. Results saved to {output_file}")
                
                # Memory management - explicitly clean up dataframes to free memory
                del df_m1, df, sampled_df
                if 'valid_df' in locals():
                    del valid_df
                if 'valid_for_accuracy' in locals():
                    del valid_for_accuracy
                import gc
                gc.collect()
                
                print("-" * 80)
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
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