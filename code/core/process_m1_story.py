import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.util import convert_to_ordinal_text
from utils.util import apply_balanced_random
from collections import Counter

# Constants
RANDOM_SEED = 2025
STORY_SUBSAMPLE_SIZE = 1000
SENTENCE_COUNT_TARGET = 5
INPUT_DATA_PATH = "./data/step1/"
OUTPUT_DATA_PATH = "./data/step2/input/"

# Set seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Define models
models = ["ChatGPT", "Claude", "Grok", "Gemini", "Llama", "Deepseek"]
# models = ["ChatGPT"]

class TextProcessor:
    """Process text data and count sentences."""
    
    def __init__(self):
        self._ensure_nltk_resources()
    
    def _ensure_nltk_resources(self):
        """Ensure required NLTK resources are available."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def count_sentences(self, text):
        """Count the number of sentences in a text."""
        if not isinstance(text, str):
            return 0
        return len(sent_tokenize(text))


def load_and_process_model_data(model_name, text_processor):
    """Load and process data for a specific model."""
    print(f"Processing {model_name} data...")
    
    try:
        # Load the CSV file
        file_path = f"{INPUT_DATA_PATH}step1_generate_stories_{model_name}.csv"
        df = pd.read_csv(file_path, sep="|")
        
        # Clean data: Remove '#' from the beginning of output if present
        df['step1_m1_output_sentence_only'] = df['step1_m1_output_sentence_only'].apply(
            lambda x: x[1:].lstrip() if isinstance(x, str) and x.startswith('#') else x
        )
        
        # Add model metadata
        df['model'] = model_name
        df["step1_m1_output_sent_count"] = df["step1_m1_output_sentence_only"].apply(text_processor.count_sentences)

        return df
        
    except Exception as e:
        print(f"Error processing {model_name}: {e}")
        return pd.DataFrame()


def prepare_model_subsample(model_df, model_name, subsample_size):
    """Prepare a subsample of stories for a specific model."""
    # Check if we have more than subsample_size stories for this model
    if len(model_df) > subsample_size:
        # Randomly sample subsample_size stories
        model_df = model_df.sample(subsample_size, random_state=RANDOM_SEED)
    
    # Print average sentence length
    avg_length = np.mean([len(x) for x in model_df['step1_m1_output_sentence_only'].dropna().tolist()])
    print(f"Average length of output for {model_name}: {avg_length:.2f} characters")

    # Convert numbers to ordinal text and create prompts
    model_df["step2_random_sent_num_ordinal_text"] = convert_to_ordinal_text(model_df, 'step2_random_sent_num')
    model_df["step2_marked_text_input_nth_sentence"] = model_df.apply(
        lambda row: f"revise only the {row['step2_random_sent_num_ordinal_text']} sentence in your own unique way, answer with only the revised sentence: {row['step1_m1_output_sentence_only']}",
        axis=1
    )
    
    # Save the sampled results
    output_path = f"{OUTPUT_DATA_PATH}mirror_test_results_add_random_result_exactly_5_sentences_{model_name}.csv"
    model_df.to_csv(output_path, index=False, sep='|')
    print(f"Saved sample of stories with exactly 5 sentences for {model_name}")
    
    return model_df


def analyze_most_frequent_sentences(df):
    """
    Analyze and display the most frequent values of step1_m1_output_sentence_only per model.
    """
    print("\n=== Most Frequent Sentences by Model ===")
    duplicate_found = False
    
    for model_name in models:
        model_df = df[df['model'] == model_name]
        
        if model_df.empty:
            print(f"\nNo data available for {model_name}")
            continue
            
        # Get the sentence frequencies
        sentence_counter = Counter(model_df['step1_m1_output_sentence_only'].dropna())
        
        if not sentence_counter:
            print(f"\nNo valid sentences found for {model_name}")
            continue
            
        # Get the most common sentence and its frequency
        most_common_sentence, frequency = sentence_counter.most_common(1)[0]
        
        # Calculate the percentage
        percentage = (frequency / len(model_df)) * 100
        
        # Check if any duplicate sentences (frequency > 1)
        has_duplicates = any(freq > 1 for _, freq in sentence_counter.items())
        if has_duplicates:
            duplicate_found = True
        
        print(f"\n{model_name}:")
        print(f"Most frequent sentence: \"{most_common_sentence}\"")
        
        # Highlight frequencies greater than 1
        if frequency > 1:
            print(f"Frequency: {frequency} occurrences ({percentage:.2f}% of {len(model_df)} samples) ⚠️ DUPLICATE DETECTED")
        else:
            print(f"Frequency: {frequency} occurrences ({percentage:.2f}% of {len(model_df)} samples)")
        
        # Display top 3 sentences if there are enough unique sentences
        if len(sentence_counter) > 1:
            print(f"\nTop 3 most frequent sentences for {model_name}:")
            for i, (sentence, freq) in enumerate(sentence_counter.most_common(3), 1):
                pct = (freq / len(model_df)) * 100
                if freq > 1:
                    print(f"{i}. \"{sentence}\" - {freq} occurrences ({pct:.2f}%) ⚠️ DUPLICATE")
                else:
                    print(f"{i}. \"{sentence}\" - {freq} occurrences ({pct:.2f}%)")
    
    return duplicate_found


def main():
    """Main function to process all model data and prepare subsamples."""
    # Ensure output directory exists
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # Initialize text processor
    text_processor = TextProcessor()
    
    # Initialize an empty list to store all dataframes
    all_dfs = []
    
    # Load and process each model's dataframe
    for model_name in tqdm(models, desc="Processing models"):
        df = load_and_process_model_data(model_name, text_processor)
        if not df.empty:
            all_dfs.append(df)
    
    # Concatenate all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save the combined dataframe
    combined_output_path = f"{INPUT_DATA_PATH}all_models_combined.csv"
    combined_df.to_csv(combined_output_path, index=False, sep="|")
    
    print(f"Combined dataframe saved to {combined_output_path}")
    print(f"Combined dataframe shape: {combined_df.shape}")
    print(f"Combined dataframe columns: {combined_df.columns.tolist()}")
    
    # Display a sample of the combined data
    print("\nSample of combined dataframe:")
    print(combined_df.head())
    
    # Display sentence count distribution by model
    sentence_counts = combined_df.groupby(["model", "step1_m1_output_sent_count"]).size().unstack(fill_value=0)
    print("\nSentence count distribution by model:")
    print(sentence_counts)
    
    # Analyze and display the most frequent sentences by model
    analyze_most_frequent_sentences(combined_df)
    
    # Filter for samples with exactly SENTENCE_COUNT_TARGET sentences
    df_target_sentences = combined_df[combined_df.step1_m1_output_sent_count == SENTENCE_COUNT_TARGET]
    
    # Assign a random sentence number (1-5) to each row
    df_target_sentences = apply_balanced_random(
        df_target_sentences, 
        count_column="step1_m1_output_sent_count",
        output_column="step2_random_sent_num", 
        random_state=RANDOM_SEED
    )
    
    # Process and subsample each model
    subsampled_dfs = []
    for model_name in models:
        model_df = df_target_sentences[df_target_sentences['model'] == model_name]
        if not model_df.empty:
            subsampled_model_df = prepare_model_subsample(model_df, model_name, STORY_SUBSAMPLE_SIZE)
            subsampled_dfs.append(subsampled_model_df)
    
    # Recombine the subsampled dataframes
    df_target_sentences_subsampled = pd.concat(subsampled_dfs, ignore_index=True)
    
    # Save the combined sampled results
    output_path = f"{OUTPUT_DATA_PATH}mirror_test_results_add_random_result_exactly_{SENTENCE_COUNT_TARGET}_sentences_combined.csv"
    df_target_sentences_subsampled.to_csv(output_path, index=False, sep='|')
    
    print(f"\nNumber of samples with exactly {SENTENCE_COUNT_TARGET} sentences after subsampling: {len(df_target_sentences_subsampled)}")
    print(f"Distribution across models: {df_target_sentences_subsampled['model'].value_counts().to_dict()}")
    print(f"Distribution of random sentence numbers: {df_target_sentences_subsampled['step2_random_sent_num'].value_counts().to_dict()}")
    
    # Analyze most frequent sentences in the subsampled dataframe
    print("\n=== Most Frequent Sentences by Model (Subsampled Data) ===")
    duplicates_found = analyze_most_frequent_sentences(df_target_sentences_subsampled)
    
    # Print summary of duplicate analysis
    print("\n" + "="*50)
    if duplicates_found:
        print("⚠️ SUMMARY: Duplicate sentences were detected in the dataset!")
        print("This may affect the validity of your mirror test analysis.")
        print("Consider examining these duplicates further or filtering them out.")
    else:
        print("✅ SUMMARY: No duplicate sentences were detected in the dataset.")
        print("All sentences appear to be unique across all models.")
    print("="*50)


if __name__ == "__main__":
    main()