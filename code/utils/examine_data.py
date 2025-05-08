import os
import random
import pandas as pd
import argparse
import sys
from io import StringIO
from datetime import datetime

# Set output file template
OUTPUT_FILE_TEMPLATE = f"./data/step3/output_processed/mirror_test_results_{{}}_{{}}.csv"

# Set random seed to 2025 as requested
random.seed(2025)

# Models list
models = ["ChatGPT", "Claude", "Grok", "Gemini", "Llama", "Deepseek"]

def examine_input_data():
    """
    Examine the mirror test input data files and save detailed examples to output/other/examples.txt.
    Shows 10 random examples for each model pair with step input and output.
    """
    # Fixed output file path
    output_file = "output/other/examples.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create a StringIO object to capture all output
    output_buffer = StringIO()
    
    # Redirect stdout to our buffer
    original_stdout = sys.stdout
    sys.stdout = output_buffer
    
    print(f"Examining mirror test input data")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Random seed: 2025")
    
    # Find all CSV files in the output directory
    input_files = []
    for m1 in models:
        for m2 in models:
            file_path = OUTPUT_FILE_TEMPLATE.format(m1, m2)
            if os.path.exists(file_path):
                input_files.append((m1, m2, file_path))
    
    print(f"Found {len(input_files)} input files")
    
    if not input_files:
        print("No input files found!")
        sys.stdout = original_stdout
        with open(output_file, 'w') as f:
            f.write(output_buffer.getvalue())
        print(f"Output saved to: {output_file}")
        return
    
    # Print summary of available files
    print("\nAvailable model pairs:")
    for m1, m2, _ in input_files:
        print(f"- {m1} vs {m2}")
    
    # Examine all files
    print(f"\nExamining all {len(input_files)} model pair files, 10 examples each:")
    
    for m1, m2, file_path in input_files:
        print(f"\n{'='*80}")
        print(f"Examining {m1} vs {m2}:")
        
        try:
            # Load the input CSV file
            df = pd.read_csv(file_path, sep='|')
            
            # Print basic stats
            print(f"Total records: {len(df)}")
            print(f"Columns: {', '.join(df.columns)}")
            
            # Check if baseline column exists
            has_baseline = "step3_input_baseline_m1_output_unchanged" in df.columns
            print(f"Has baseline column: {has_baseline}")
            
            # Show 10 random examples
            num_examples_to_show = min(10, len(df))
            random_indices = random.sample(range(len(df)), num_examples_to_show)
            
            print(f"\nShowing {num_examples_to_show} random examples with full details:")
            
            for idx, row_idx in enumerate(random_indices):
                row = df.iloc[row_idx]
                
                print(f"\n{'#'*100}")
                print(f"EXAMPLE {idx+1}:")
                print(f"{'#'*100}")
                
                # Display source_model and target_model first if they exist
                if "source_model" in row and "target_model" in row:
                    print(f"\n{'*'*60}")
                    print(f"SOURCE MODEL: {row['source_model']} | TARGET MODEL: {row['target_model']}")
                    print(f"{'*'*60}")
                elif "source_model" in row:
                    print(f"\n{'*'*60}")
                    print(f"SOURCE MODEL: {row['source_model']}")
                    print(f"{'*'*60}")
                elif "target_model" in row:
                    print(f"\n{'*'*60}")
                    print(f"TARGET MODEL: {row['target_model']}")
                    print(f"{'*'*60}")
                
                # Display step2_random_sent_num prominently
                if "step2_random_sent_num" in row:
                    print(f"\n{'*'*60}")
                    print(f"RANDOM SENTENCE NUMBER: {row['step2_random_sent_num']}")
                    print(f"{'*'*60}")
                
                # Organize columns by step for clearer output
                step1_columns = [
                    "step1_story_prompt_with_prefix",
                    "step1_m1_output_sentence_only",
                ]
                
                step2_columns = [
                    "step2_output_nth_sentence_message_only",
                    "step2_output",
                    "step2_random_sent_num",
                    "step2_marked_text_input_nth_sentence",
                ]
                
                step3_columns = [
                    "step3_input",
                    "step3_output",
                    "step3_output_message_only",
                ]
                
                # Add other columns that might be present
                other_columns = [
                    "source_model",
                    "target_model",
                ]
                
                # Always show baseline column if it exists
                if has_baseline:
                    step3_columns.append("step3_input_baseline_m1_output_unchanged")
                
                # Group columns by step for more organized output
                for step_name, columns in [
                    ("STEP 1: STORY GENERATION", step1_columns),
                    ("STEP 2: SENTENCE REVISION", step2_columns),
                    ("STEP 3: STRANGE PART IDENTIFICATION", step3_columns),
                ]:
                    print(f"\n{'-'*40}")
                    print(f"{step_name}:")
                    print(f"{'-'*40}")
                    
                    for column in columns:
                        # Skip columns that are already displayed prominently
                        if column in ["step2_random_sent_num", "source_model", "target_model"]:
                            continue
                            
                        if column in row:
                            print(f"\n{'-'*30}")
                            print(f"{column}:")
                            print(f"{'-'*30}")
                            print(row[column])
                        else:
                            print(f"\n{'-'*30}")
                            print(f"{column}: NOT FOUND")
                            print(f"{'-'*30}")
                
        except Exception as e:
            print(f"Error examining file {file_path}: {e}")
    
    # Restore stdout and save to file
    sys.stdout = original_stdout
    
    # Save the captured output to the fixed file path
    with open(output_file, 'w') as f:
        f.write(output_buffer.getvalue())
    
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    # Run the data examination with fixed random seed of 2025
    examine_input_data()