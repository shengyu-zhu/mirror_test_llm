import os
import pandas as pd
import numpy as np
from plotnine import *
import argparse
from tqdm import tqdm

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Visualize sentence length distributions')
    parser.add_argument('--baseline', action='store_true', help='Use baseline data from _m1_output_unchanged directory')
    args = parser.parse_args()
    
    # Paths
    OUTPUT_PATH = "./output/"
    FIGURE_PATH = os.path.join(OUTPUT_PATH, "figure/effect_of_length_of_sentences/")
    os.makedirs(FIGURE_PATH, exist_ok=True)

    # Data directory for sentence count results
    SENTENCE_COUNT_DIR = "./data/sentence_count_result/"
    
    # Find all result files with different sentence counts
    print("\n===== Analyzing Results =====")
    print("Finding result files for different sentence counts...")
    
    # Track the data for each sentence count
    sentence_length_data = []
    
    # Sentence counts to look for
    sentence_counts = [2, 3, 5, 7, 10, 13, 15, 18, 20]
    
    # Process each sentence count
    for sentence_count in tqdm(sentence_counts, desc="Processing sentence counts"):
        # Find files for this sentence count
        matching_files = [f for f in os.listdir(SENTENCE_COUNT_DIR) if f.startswith("mirror_test_results_") and f"_{sentence_count}_sentences.csv" in f]
        
        for file_name in matching_files:
            file_path = os.path.join(SENTENCE_COUNT_DIR, file_name)
            
            try:
                # Read the CSV with pipe separator
                df = pd.read_csv(file_path, sep="|")
                
                # Ensure numeric values
                df['step2_random_sent_num'] = pd.to_numeric(df['step2_random_sent_num'], errors='coerce')
                df['step3_output_int'] = pd.to_numeric(df['step3_output_int'], errors='coerce')
                
                # Add sentence count to the dataframe
                df['sentence_count'] = sentence_count
                
                # Add to our collection
                sentence_length_data.append(df)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    # Combine all data
    if not sentence_length_data:
        print("No data files found or processed successfully.")
        return
        
    print("Combining data from all sentence counts...")
    combined_df = pd.concat(sentence_length_data, ignore_index=True)
    
    # Calculate distributions by sentence count
    # 1. Distribution of actual modified sentences (step2_random_sent_num)
    actual_distributions = []
    
    print("Calculating distributions for each sentence count...")
    
    # Process each sentence count
    for sentence_count in sentence_counts:
        subset = combined_df[combined_df['sentence_count'] == sentence_count]
        
        if subset.empty:
            print(f"No data for sentence count {sentence_count}")
            continue
            
        # Get max sentence number (up to 20 for extended range)
        max_sentence = min(20, sentence_count)
        
        # Filter to only relevant sentence numbers (1 to max_sentence)
        valid_subset = subset[
            subset['step2_random_sent_num'].between(1, max_sentence) & 
            subset['step3_output_int'].between(1, max_sentence)
        ]
        
        if valid_subset.empty:
            continue
            
        # 1. Distribution of actual modified sentences
        actual_dist = (
            valid_subset['step2_random_sent_num'].value_counts(normalize=True) * 100
        ).reset_index()
        actual_dist.columns = ['step_num', 'percentage']
        actual_dist['step_num'] = actual_dist['step_num'].astype(int)
        actual_dist['sentence_count'] = sentence_count
        actual_dist['source'] = 'Actual Modified Sentence'
        actual_distributions.append(actual_dist)
    
    # Combine all distributions
    if not actual_distributions:
        print("No valid distributions could be calculated.")
        return
        
    distribution_df = pd.concat(actual_distributions, ignore_index=True)
    
    # Generate the Actual Modified Sentence plot
    print("Generating Actual Modified Sentence plot...")
    
    # Create plot
    plot = (
        ggplot(distribution_df, aes(x='step_num', y='percentage', color='factor(sentence_count)', group='factor(sentence_count)'))
        + geom_line(size=1.2)
        + geom_point(size=3)
        + scale_x_continuous(breaks=[1, 3, 5, 7, 10, 15, 20], limits=[0.5, 20.5])
        + scale_y_continuous(
            limits=[0, 100],
            breaks=[0, 20, 40, 60, 80, 100],
            labels=lambda x: [f"{v}%" for v in x]
        )
        + scale_color_brewer(type='qual', palette='Dark2')
        + theme_minimal()
        + theme(
            legend_position='bottom', 
            figure_size=(1.618 * 5, 5),
            panel_grid_minor=element_blank(),
            panel_grid_major_y=element_line(color='lightgray'),
            panel_grid_major_x=element_blank(),
            text=element_text(size=12),
            axis_text=element_text(size=11),
            plot_title=element_text(size=16, face="bold"),
            legend_title=element_text(size=12),
            legend_text=element_text(size=11)
        )
        + labs(
            title='Actual Modified Sentence by Story Length',
            x='Sentence Number (1-20)',
            y='Percentage',
            color='Sentence Count'
        )
    )
    
    # Save plot
    plot_path = os.path.join(FIGURE_PATH, "distribution_Actual_Modified_Sentence_by_sentence_count.png")
    plot.save(plot_path, dpi=600, width=10, height=6)
    print(f"Saved Actual Modified Sentence plot to: {plot_path}")

if __name__ == "__main__":
    main()