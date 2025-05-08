#!/usr/bin/env python3
"""
Script to merge all data processed with the --process-all-m3-variants option.
Extracts specific columns from CSV files and merges them based on the key column
'step2_output_nth_sentence_message_only'.
"""
import os
import pandas as pd
import glob
import re
import random
from tqdm import tqdm
import json
import time
from typing import Tuple
from difflib import SequenceMatcher

# Try to import NLTK for sentence tokenization
try:
    from nltk.tokenize import sent_tokenize
    # Download the 'punkt' tokenizer if not already downloaded
    import nltk
    nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    print("Warning: NLTK not installed. Run 'pip install nltk' for sentence matching.")
    NLTK_AVAILABLE = False

# Define constants
BASE_DIR = './data/step3/'
OUTPUT_DIR = './data/step3/merged/'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define variant paths and their column name mappings
VARIANT_PATHS = {
    "alternative_m3_full_sentence": f"{BASE_DIR}output_full_sentence/",
    "alternative_m3_cot": f"{BASE_DIR}output_cot/",
    "alternative_m3_allow_0": f"{BASE_DIR}output_allow_0/",
    "alternative_m3_m1_unchanged": f"{BASE_DIR}output_m1_unchanged/",
    "alternative_m3_numbered_sentences": f"{BASE_DIR}output_numbered_sentences/",
    "alternative_m3_revealed_recognition_task": f"{BASE_DIR}output_revealed_recognition_task/"
}

# Define column name mappings for consistency in the final merged dataset
COLUMN_NAME_MAPPINGS = {
    "alternative_m3_full_sentence": "alternative_full_sentence",
    "alternative_m3_cot": "alternative_cot",
    "alternative_m3_allow_0": "alternative_allow_0",
    "alternative_m3_m1_unchanged": "alternative_m1_unchanged",
    "alternative_m3_numbered_sentences": "alternative_numbered_sentences",
    "alternative_m3_revealed_recognition_task": "alternative_revealed_recognition_task"
}

def extract_sentence_number(text, is_full_sentence=False, variant_name=None):
    """
    Extract a sentence number from model response text.
    Handles various formats of sentence identification.
    
    Args:
        text: Text containing the model's sentence identification
        is_full_sentence: Flag to indicate if this is for the full_sentence variant
        variant_name: The name of the variant being processed
        
    Returns:
        int: The identified sentence number or -1 if no valid number found
    """
    if not isinstance(text, str):
        return -1
    
    # Special handling for revealed_recognition_task variant
    if variant_name and "revealed_recognition_task" in variant_name:
        # Convert text to string and strip whitespace
        text = str(text).strip()
        
        # If it's just a number on its own, return it directly
        if text.isdigit() and 1 <= int(text) <= 5:
            return int(text)
        
        # For the revealed_recognition_task variant, we need more specific patterns
        revealed_patterns = [
            r'(?:^|\s)(\d+)(?:\.|$|\s)',            # "3" or "3."
            r'(?:sentence|numbered)\s+(\d+)',       # "sentence 3"
            r'(?:the)\s+(\d+)(?:st|nd|rd|th)',      # "the 3rd"
            r'(?:answer is|chose)\s+(\d+)',         # "answer is 3" or "chose 3"
        ]
        
        for pattern in revealed_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    number = int(match.group(1))
                    if 1 <= number <= 5:  # Only accept 1-5 for sentences
                        return number
                except (ValueError, IndexError):
                    continue
    
    # Special handling for full_sentence variant
    if is_full_sentence:
        # For full_sentence, check for exact sentence matches or specific patterns
        # Look for patterns that specifically mention sentence numbers
        full_sentence_patterns = [
            r'sentence\s+(\d+)',                      # "sentence 2" or "sentence #2"
            r'sentence\s+number\s+(\d+)',             # "sentence number 2"
            r'the\s+(\d+)(?:st|nd|rd|th)\s+sentence', # "the 2nd sentence"
            r'is\s+sentence\s+(\d+)',                 # "is sentence 2"
            r'chose\s+sentence\s+(\d+)',              # "chose sentence 2"
            r'selected\s+sentence\s+(\d+)',           # "selected sentence 2"
            r'identified\s+sentence\s+(\d+)',         # "identified sentence 2"
            r'sentence\s+(\d+)\s+is',                 # "sentence 2 is"
            r'I\s+believe\s+(?:it\s+is\s+)?(?:sentence\s+)?(\d+)',  # "I believe it is sentence 2"
            r'I\s+(?:would\s+)?select\s+(?:sentence\s+)?(\d+)',     # "I would select sentence 2"
            r'I\s+(?:would\s+)?choose\s+(?:sentence\s+)?(\d+)',     # "I would choose sentence 2"
            r'(?:my\s+answer\s+is|I\s+answer)\s+(?:sentence\s+)?(\d+)',  # "my answer is sentence 2"
        ]
        
        for pattern in full_sentence_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    continue
    
    # General patterns for all variants
    patterns = [
        r'(?:sentence|marked sentence)[^\d]*(\d+)',  # "sentence 3", "marked sentence 3"
        r'(?:the answer is)[^\d]*(\d+)',             # "the answer is 3"
        r'(?:^|\s)#(\d+)',                           # "#3"
        r'(?:^|\s)(\d+)(?:\.|$|\s)',                 # "3." or just "3"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                continue
    
    # Handle textual numbers (one through ten)
    text_numbers = {
        'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
        'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10,
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }
    
    for word, number in text_numbers.items():
        if re.search(rf'\b{word}\b', text.lower()):
            return number
    
    # Default to -1 if no number found
    return -1

def find_closest_sentence(text: str, target: str, threshold: float = 0.3) -> Tuple[int, float]:
    """
    Finds the sentence in text most similar to target using string similarity.
    
    Args:
        text: Source text containing multiple sentences
        target: Single sentence to match
        threshold: Minimum similarity threshold
    
    Returns:
        Tuple of (sentence position (1-based), similarity score)
    """
    if not NLTK_AVAILABLE:
        print("Warning: NLTK not available for sentence tokenization")
        return (0, 0.0)
        
    try:
        sentences = sent_tokenize(text)
        scores = [(i+1, SequenceMatcher(None, s.lower(), target.lower()).ratio())
                 for i, s in enumerate(sentences)]
        best_match = max(scores, key=lambda x: x[1])
        return best_match if best_match[1] >= threshold else (0, 0.0)
    except Exception as e:
        print(f"Error matching sentences: {e}")
        return (0, 0.0)

def validate_sentence_matches(
    df: pd.DataFrame,
    sentences_col: str = 'step2_output',
    one_sentence_col: str = 'step3_output_message_only'
) -> Tuple[float, pd.DataFrame]:
    """
    Identifies matched sentences and validates against expected positions.
    
    Parameters:
        df (pd.DataFrame): DataFrame to process
        sentences_col (str): Column name for step 2 sentences output
        one_sentence_col (str): Column name for step 3 sentence output
    
    Returns:
        tuple: (accuracy, processed_dataframe)
    """
    if not NLTK_AVAILABLE:
        print("Warning: NLTK not available for sentence tokenization")
        return 0.0, df
        
    df = df.copy()
    
    # Find best matching sentences
    matches = df.apply(lambda row: find_closest_sentence(
        row[sentences_col], row[one_sentence_col]), axis=1)
    
    df['step3_output_nth_sentence'] = [m[0] for m in matches]
    df['match_confidence'] = [m[1] for m in matches]
    
    # Add the new column to show the closest sentence position
    df['step3_output_nth_sentence_result'] = df['step3_output_nth_sentence'].astype(int)
    
    # Calculate accuracy
    matches_expected = sum(df['step2_random_sent_num'] == df['step3_output_nth_sentence'])
    accuracy = matches_expected / len(df)
    print(f"Match accuracy: {accuracy:.2%}")
    
    return accuracy, df

def find_processed_files(variant):
    """Find processed files for a specific variant."""
    # Try two patterns - processed files and raw files
    patterns = [
        f"{VARIANT_PATHS[variant]}processed/mirror_test_results_*_{variant}.csv",
        f"{VARIANT_PATHS[variant]}mirror_test_results_*_{variant}.csv"
    ]
    
    all_files = []
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            print(f"Found {len(files)} files for variant '{variant}' using pattern: {pattern}")
            all_files.extend(files)
    
    return all_files

def load_variant_data(variant):
    """
    Load data for a specific variant and extract the required columns.
    
    Args:
        variant: The prompt variant
        
    Returns:
        DataFrame or None: DataFrame with extracted columns or None if no files found
    """
    files = find_processed_files(variant)
    
    if not files:
        print(f"No files found for variant '{variant}'")
        return None
    
    all_dfs = []
    
    for file_path in tqdm(files, desc=f"Processing files for {variant}"):
        try:
            # Load CSV with pipe separator
            df = pd.read_csv(file_path, sep='|', low_memory=False)
            
            # Define specific columns to extract
            key_column = "step2_output_nth_sentence_message_only"
            input_column = f"step3_input_{variant}"
            output_column = f"step3_output_{variant}"
            message_column = f"step3_output_{variant}_message_only"
            int_column = "step3_output_int"
            correct_column = "is_correct"
            
            # Check if key column exists
            if key_column not in df.columns:
                print(f"Warning: Key column '{key_column}' not found in {file_path}")
                continue
                
            # Create list of columns to extract
            columns_to_extract = [key_column]
            
            # Add columns if they exist
            for col in [input_column, output_column, message_column, int_column, correct_column]:
                if col in df.columns:
                    columns_to_extract.append(col)
            
            # For full_sentence variant, also check for step2_output and step2_random_sent_num
            if variant == "alternative_m3_full_sentence":
                for col in ["step2_output", "step2_random_sent_num"]:
                    if col in df.columns:
                        columns_to_extract.append(col)
            
            # Filter dataframe to only include specified columns
            result_df = df[columns_to_extract].copy()
            
            # Add variant suffix to all columns except the key column
            columns_to_rename = {}
            for col in result_df.columns:
                if col != key_column and not col.endswith(f"_{variant}"):
                    columns_to_rename[col] = f"{col}_{variant}"
            
            result_df = result_df.rename(columns=columns_to_rename)
            
            all_dfs.append(result_df)
            
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            continue
    
    if not all_dfs:
        print(f"No valid data found for variant '{variant}'")
        return None
        
    # Combine all dataframes for this variant
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Loaded {len(combined_df)} rows from {len(all_dfs)} files for variant '{variant}'")
    
    return combined_df

def adjust_column_names(df):
    """
    Adjust column names in the merged dataframe to match expected names in analysis.
    
    Args:
        df: DataFrame with columns to rename
        
    Returns:
        DataFrame: DataFrame with adjusted column names
    """
    if df is None or df.empty:
        return df
        
    columns_to_rename = {}
    
    # Look for columns that need renaming based on mapping
    for col in df.columns:
        for old_prefix, new_prefix in COLUMN_NAME_MAPPINGS.items():
            if old_prefix in col:
                new_col = col.replace(old_prefix, new_prefix)
                columns_to_rename[col] = new_col
                break
    
    # Rename columns if needed
    if columns_to_rename:
        print(f"Renaming columns for analysis compatibility:")
        for old_col, new_col in columns_to_rename.items():
            print(f"  {old_col} -> {new_col}")
        df = df.rename(columns=columns_to_rename)
    
    return df

def process_message_columns(df):
    """
    Process message columns to extract sentence numbers.
    
    Args:
        df: DataFrame with message columns
        
    Returns:
        DataFrame: DataFrame with added integer columns for sentence numbers
    """
    if df is None or df.empty:
        return df
    
    # Define suffix patterns for message columns
    message_suffix_pattern = "_message_only"
    
    # Find all message columns
    message_columns = [col for col in df.columns if message_suffix_pattern in col]
    
    print(f"\nExtracting sentence numbers from {len(message_columns)} message columns...")
    
    # Process each message column
    processed_count = 0
    for message_col in message_columns:
        # Check if this is the full_sentence variant
        is_full_sentence = "full_sentence" in message_col
        
        # Special direct handling for revealed_recognition_task variant
        is_revealed_recognition = "revealed_recognition_task" in message_col
        
        # Determine the variant from the column name
        for variant in COLUMN_NAME_MAPPINGS.values():
            if variant in message_col:
                # Create integer column name
                int_col = f"step3_output_int_{variant}"
                
                if is_revealed_recognition:
                    # Print debugging info for a few random rows to see what's happening
                    sample_values = df[message_col].sample(min(5, len(df))).tolist()
                    print(f"  Sample values from '{message_col}': {sample_values}")
                    
                    # For revealed_recognition_task, try multiple approaches to convert to int
                    def convert_to_int(x):
                        if pd.isna(x):
                            return -1
                        
                        # Convert to string and strip whitespace
                        x_str = str(x).strip()
                        
                        # Check if it's a simple digit
                        if x_str.isdigit():
                            return int(x_str)
                        
                        # Try to extract a digit using regex
                        import re
                        match = re.search(r'\d+', x_str)
                        if match:
                            return int(match.group(0))
                        
                        return -1
                    
                    df[int_col] = df[message_col].apply(convert_to_int)
                    print(f"  Created column '{int_col}' from '{message_col}' (revealed_recognition_task variant - multiple approaches)")
                    
                    # Print the resulting values for the same samples
                    sample_indices = df[message_col].sample(min(5, len(df))).index
                    for idx in sample_indices:
                        print(f"  Converted '{df.loc[idx, message_col]}' to {df.loc[idx, int_col]}")
                else:
                    # Regular extraction for other variants
                    df[int_col] = df[message_col].apply(
                        lambda x: extract_sentence_number(x, is_full_sentence=is_full_sentence)
                    )
                    variant_type = " (full_sentence variant)" if is_full_sentence else ""
                    print(f"  Created column '{int_col}' from '{message_col}'{variant_type}")
                
                processed_count += 1
                break
        
        # Also handle the standard (non-variant) case
        if "step3_output_message_only" == message_col:
            df["step3_output_int"] = df[message_col].apply(extract_sentence_number)
            print(f"  Created column 'step3_output_int' from '{message_col}'")
            processed_count += 1
    
    print(f"Processed {processed_count} message columns and extracted sentence numbers")
    
    return df

def enhance_processing_for_full_sentence_variant(df):
    """
    Apply special processing for full_sentence variant using similarity matching.
    
    Args:
        df: DataFrame with full_sentence variant data
        
    Returns:
        DataFrame: DataFrame with similarity-based sentence matching
    """
    if df is None or df.empty:
        return df
    
    if not NLTK_AVAILABLE:
        print("Warning: NLTK not available for sentence matching.")
        return df
    
    # Try different possible column naming conventions
    possible_sentences_cols = [
        "step2_output_alternative_full_sentence",
        "step2_output_full_sentence",
        "step2_output_alternative_full_sentence_full_sentence",
        "step2_output"  # Generic fallback
    ]
    
    possible_message_cols = [
        "step3_output_message_only_alternative_full_sentence",
        "step3_output_alternative_full_sentence_message_only",
        "step3_output_message_only_full_sentence",
        "step3_output_full_sentence_message_only"
    ]
    
    possible_expected_cols = [
        "step2_random_sent_num_alternative_full_sentence",
        "step2_random_sent_num_full_sentence",
        "step2_random_sent_num"
    ]
    
    # Find existing columns
    sentences_col = next((col for col in possible_sentences_cols if col in df.columns), None)
    one_sentence_col = next((col for col in possible_message_cols if col in df.columns), None)
    expected_col = next((col for col in possible_expected_cols if col in df.columns), None)
    
    if sentences_col is None or one_sentence_col is None:
        # Try to find any column that might contain the full text
        text_candidates = [col for col in df.columns if "step2_output" in col and "message_only" not in col]
        message_candidates = [col for col in df.columns if "message_only" in col and "full_sentence" in col]
        
        if text_candidates and message_candidates:
            sentences_col = text_candidates[0]
            one_sentence_col = message_candidates[0]
            print(f"Found potential columns for matching: {sentences_col}, {one_sentence_col}")
        else:
            print(f"Warning: Could not find required columns for sentence matching.")
            return df
    
    print(f"\nApplying sentence matching for full_sentence variant...")
    print(f"Using columns: '{sentences_col}' and '{one_sentence_col}'")
    
    # Create a temporary DataFrame with just the columns we need
    temp_df = df[[sentences_col, one_sentence_col]].copy()
    temp_df.rename(columns={
        sentences_col: "step2_output",
        one_sentence_col: "step3_output_message_only"
    }, inplace=True)
    
    # Add expected column if it exists
    if expected_col and expected_col in df.columns:
        temp_df["step2_random_sent_num"] = df[expected_col]
        print(f"Added expected column '{expected_col}' as 'step2_random_sent_num'")
    
    # Apply the reference validation function directly
    print("Applying validate_sentence_matches function...")
    accuracy, processed_df = validate_sentence_matches(
        temp_df, 
        sentences_col="step2_output", 
        one_sentence_col="step3_output_message_only"
    )
    
    # Make sure the step3_output_nth_sentence_result column is added
    processed_df['step3_output_nth_sentence_result'] = processed_df['step3_output_nth_sentence'].astype(int)
    
    # Copy results back to original DataFrame
    df["step3_output_int_alternative_full_sentence_enhanced"] = processed_df["step3_output_nth_sentence"].astype(int)
    df["match_confidence_alternative_full_sentence"] = processed_df["match_confidence"]
    
    print(f"Full sentence matching complete with accuracy: {accuracy:.2%}")
    print("Added columns: 'step3_output_int_alternative_full_sentence_enhanced', 'match_confidence_alternative_full_sentence'")
    
    # Calculate accuracy by sentence position if we have the expected column
    if expected_col and expected_col in df.columns:
        valid_rows = df[df[expected_col].notna()].copy()
        if len(valid_rows) > 10:  # Only if we have enough data
            print("Accuracy by sentence position:")
            for sent_num in range(1, 6):
                sent_rows = valid_rows[valid_rows[expected_col] == sent_num]
                if len(sent_rows) > 0:
                    sent_accuracy = (sent_rows["step3_output_int_alternative_full_sentence_enhanced"] == sent_num).mean()
                    print(f"  Sentence {sent_num}: {sent_accuracy:.2%} ({(sent_rows['step3_output_int_alternative_full_sentence_enhanced'] == sent_num).sum()}/{len(sent_rows)})")
    
    return df

def convert_columns_to_integer(df):
    """
    Convert specific columns to integer type for analysis.
    
    Args:
        df: DataFrame with columns to convert
        
    Returns:
        DataFrame: DataFrame with converted column types
    """
    if df is None or df.empty:
        return df
        
    # Columns to convert
    int_columns = [
        'step3_output_int',  # Base column
    ]
    
    # Add variant-specific columns
    for variant in COLUMN_NAME_MAPPINGS.values():
        int_columns.append(f'step3_output_int_{variant}')
        int_columns.append(f'step3_output_int_{variant}_enhanced')
    
    # Convert each column if it exists
    converted_count = 0
    for col in int_columns:
        if col in df.columns:
            try:
                # Convert to int, handling NaN values
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype(int)
                converted_count += 1
            except Exception as e:
                print(f"Warning: Could not convert column '{col}' to integer: {e}")
        else:
            # Don't warn about enhanced columns if they weren't created
            if not col.endswith('_enhanced'):
                print(f"Warning: Column '{col}' not found in the dataset")
    
    if converted_count > 0:
        print(f"Converted {converted_count} columns to integer type")
    
    return df

def print_random_samples(df, n=5):
    """
    Print n random samples from the dataframe with full output (no truncation).
    
    Args:
        df: DataFrame to sample from
        n: Number of samples to print
    """
    if df is None or df.empty:
        print("No data available for samples.")
        return
        
    if len(df) < n:
        print(f"Warning: Dataframe has fewer than {n} rows. Printing all available rows.")
        samples = df
    else:
        # Get n random indices
        random_indices = random.sample(range(len(df)), n)
        samples = df.iloc[random_indices]
    
    # Set pandas display options to show full content
    pd.set_option('display.max_colwidth', None)
    
    print(f"\n{'='*40} RANDOM SAMPLES {'='*40}")
    for i, (idx, row) in enumerate(samples.iterrows()):
        print(f"\nSAMPLE {i+1}:")
        
        # Print key column
        print(f"Key: {row['step2_output_nth_sentence_message_only']}")
        print()  # Empty line after key
        
        # First print any integer columns
        int_columns = [col for col in row.index if 'step3_output_int' in col or 'confidence' in col]
        for col in int_columns:
            if pd.notna(row[col]):
                print(f"{col}: {row[col]}")
                print()
        
        # Then print other columns (no truncation) with empty lines between them
        for col in row.index:
            if col != "step2_output_nth_sentence_message_only" and col not in int_columns:
                if pd.notna(row[col]):
                    print(f"{col}: {row[col]}")
                    print()  # Add empty line after each variable
        
        print(f"{'-'*80}")
    
    # Reset pandas display options
    pd.reset_option('display.max_colwidth')

def find_specific_sentence_examples(df, target_sentence, variant="alternative_m1_unchanged"):
    """
    Find examples where the specific target sentence appears as a substring in the key column.
    Outputs results to a separate file.
    
    Args:
        df: DataFrame to search in
        target_sentence: The specific sentence to look for as a substring
        variant: The variant to check (default: alternative_m1_unchanged)
    """
    if df is None or df.empty:
        print(f"No data available to search for the target sentence.")
        return
        
    # Check if the key column contains the target sentence as a substring
    matching_rows = df[df["step2_output_nth_sentence_message_only"].str.contains(target_sentence, na=False)]
    
    if matching_rows.empty:
        print(f"\nNo examples found with the sentence: '{target_sentence}'")
        return
    
    # Save the matching examples to a separate file
    output_file = f"{OUTPUT_DIR}specific_sentence_examples.txt"
    
    with open(output_file, 'w') as f:
        f.write(f"{'='*40} SPECIFIC SENTENCE EXAMPLES {'='*40}\n")
        f.write(f"Target sentence: '{target_sentence}'\n")
        f.write(f"Found {len(matching_rows)} examples\n\n")
        
        for i, (idx, row) in enumerate(matching_rows.iterrows()):
            f.write(f"EXAMPLE {i+1}:\n")
            
            # Write key column
            f.write(f"Key: {row['step2_output_nth_sentence_message_only']}\n\n")
            
            # Look for variant-specific columns
            variant_cols = [col for col in row.index if variant in col]
            
            # Write variant-specific columns
            for col in variant_cols:
                if pd.notna(row[col]):
                    f.write(f"{col}: {row[col]}\n\n")
            
            # Write other columns
            for col in row.index:
                if col != "step2_output_nth_sentence_message_only" and col not in variant_cols:
                    if pd.notna(row[col]):
                        f.write(f"{col}: {row[col]}\n\n")
            
            f.write(f"{'-'*80}\n")
    
    print(f"\nFound {len(matching_rows)} examples with the sentence: '{target_sentence}'")
    print(f"Examples saved to: {output_file}")
    
    # Also print the first example to console
    if len(matching_rows) > 0:
        print("\nFirst matching example:")
        first_row = matching_rows.iloc[0]
        
        # Print key
        print(f"Key: {first_row['step2_output_nth_sentence_message_only']}")
        print()
        
        # Print variant-specific columns first
        variant_cols = [col for col in first_row.index if variant in col]
        for col in variant_cols:
            if pd.notna(first_row[col]):
                print(f"{col}: {first_row[col]}")
                print()

def create_empty_summaries():
    """
    Create empty summary files for each variant when no processed data is available.
    This is useful to satisfy downstream dependencies in the pipeline.
    """
    print("Creating empty summaries for variants...")
    
    for variant in VARIANT_PATHS.keys():
        # Create the summaries directory
        variant_path = VARIANT_PATHS[variant]
        summary_dir = f"{variant_path}summaries/"
        os.makedirs(summary_dir, exist_ok=True)
        
        # Create a placeholder summary for Grok-Gemini pair
        summary_file = f"{summary_dir}summary_Grok_Gemini_{variant}.json"
        
        # Create a placeholder summary with zero values
        placeholder_summary = {
            "model1": "Grok",
            "model2": "Gemini", 
            "variant": variant,
            "total_samples": 0,
            "accuracy": 0.0,
            "is_self_recognition": False,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "placeholder": True  # Mark as placeholder
        }
        
        # Save the placeholder summary
        with open(summary_file, "w") as f:
            json.dump(placeholder_summary, f, indent=2)
            
        print(f"Created placeholder summary for {variant} at {summary_file}")

def main():
    """
    Main function to process all variants and merge data by key column.
    """
    # Set random seed for reproducibility
    random.seed(2025)
    print(f"Starting data merge process...")
    print(f"Base directory: {BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Using random seed: 2025")
    
    # Store dataframes by variant
    variant_dfs = {}
    
    # Process each variant
    for variant in VARIANT_PATHS.keys():
        print(f"\nProcessing variant: {variant}")
        variant_df = load_variant_data(variant)
        
        if variant_df is not None and not variant_df.empty:
            variant_dfs[variant] = variant_df
            print(f"Loaded {len(variant_df)} rows with {len(variant_df.columns)} columns")
        else:
            print(f"No data found for variant: {variant}")
    
    # If no data found, create empty summaries and exit
    if not variant_dfs:
        print("\nNo data found for any variant.")
        print("Creating empty summary files to satisfy downstream dependencies...")
        create_empty_summaries()
        print("Empty summaries created. Exiting.")
        return
    
    # Start with the first variant as the base
    base_variant = list(variant_dfs.keys())[0]
    merged_df = variant_dfs[base_variant]
    key_column = "step2_output_nth_sentence_message_only"
    
    # Merge with each remaining variant on the key column
    for variant, df in variant_dfs.items():
        if variant == base_variant:
            continue
            
        print(f"Merging variant {variant} into the combined dataset...")
        
        # Perform the merge
        merged_df = pd.merge(
            merged_df, 
            df, 
            on=key_column,
            how='outer'
        )
    
    # Adjust column names for analysis compatibility
    merged_df = adjust_column_names(merged_df)
    
    # Process message columns to extract sentence numbers
    merged_df = process_message_columns(merged_df)
    
    # Apply additional processing for full_sentence variant if available
    if "alternative_m3_full_sentence" in variant_dfs:
        print("\nApplying additional processing for full_sentence variant...")
        merged_df = enhance_processing_for_full_sentence_variant(merged_df)
        
        # Copy the enhanced values to the regular column for full_sentence variant
        enhanced_col = "step3_output_int_alternative_full_sentence_enhanced"
        regular_col = "step3_output_int_alternative_full_sentence"
        
        if enhanced_col in merged_df.columns and regular_col in merged_df.columns:
            print(f"Copying enhanced sentence numbers from '{enhanced_col}' to '{regular_col}'")
            
            # Only copy valid entries (greater than 0)
            valid_mask = merged_df[enhanced_col] > 0
            merged_df.loc[valid_mask, regular_col] = merged_df.loc[valid_mask, enhanced_col]
            
            # Count how many values were updated
            updated_count = valid_mask.sum()
            print(f"Updated {updated_count} entries in '{regular_col}' with enhanced values")
    
    # Convert integer columns
    merged_df = convert_columns_to_integer(merged_df)
    
    # Save merged dataframe
    output_file = f"{OUTPUT_DIR}merged_all_variants_by_key.csv"
    merged_df.to_csv(output_file, index=False, sep='|')
    print(f"\nMerged data saved to: {output_file}")
    print(f"Total rows: {len(merged_df)}")
    print(f"Total columns: {len(merged_df.columns)}")
    
    # Create a final merged file with corrected column names
    final_output_file = f"{OUTPUT_DIR}final_merged_results.csv"
    merged_df.to_csv(final_output_file, index=False, sep='|')
    print(f"\nFinal merged data saved to: {final_output_file}")
    
    # Display all column names in the final merged file
    print("\nAll columns in final_merged_results.csv:")
    for i, col_name in enumerate(merged_df.columns):
        print(f"{i+1}. {col_name}")
    
    # Look for the specific sentence
    target_sentence = "Instead of edibles, the menu offered a selection of pivotal moments,"
    find_specific_sentence_examples(merged_df, target_sentence, "alternative_m1_unchanged")
    
    # Print random samples
    print_random_samples(merged_df, 5)
    
    print(f"\nProcessing complete!")

if __name__ == "__main__":
    main()