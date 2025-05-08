import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def process_sentence_count_data(sentence_count, sentence_count_dir, figure_path):
    """Process data for a specific sentence count and generate three figures"""
    print(f"\nProcessing data for {sentence_count} sentences...")
    
    # Create directory for this sentence count
    sentence_dir = os.path.join(figure_path, f"{sentence_count}_sentences")
    os.makedirs(sentence_dir, exist_ok=True)
    
    # Find all files for this sentence count
    matching_files = []
    for root, _, files in os.walk(sentence_count_dir):
        for file in files:
            if file.endswith(f"_{sentence_count}_sentences.csv"):
                matching_files.append(os.path.join(root, file))
    
    if not matching_files:
        print(f"No files found for {sentence_count} sentences")
        return
    
    print(f"Found {len(matching_files)} files for {sentence_count} sentences")
    
    # Load and combine data
    df_list = []
    for file_path in matching_files:
        try:
            df = pd.read_csv(file_path, sep="|", engine="python")
            # Extract model names from filename if they're not in the data
            if 'model1' not in df.columns or 'model2' not in df.columns:
                filename = os.path.basename(file_path)
                parts = filename.split('_')
                if len(parts) >= 4:
                    model1 = parts[2]
                    model2 = parts[3].split('_')[0]  # Remove the "_sentences" part
                    df['model1'] = model1
                    df['model2'] = model2
            df_list.append(df)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not df_list:
        print(f"No valid data found for {sentence_count} sentences")
        return
    
    # Combine all dataframes
    df = pd.concat(df_list, ignore_index=True)
    
    # Rename for consistency
    if 'model1' in df.columns and 'Model_1' not in df.columns:
        df['Model_1'] = df['model1']
    if 'model2' in df.columns and 'Model_2' not in df.columns:
        df['Model_2'] = df['model2']
    
    # Clean and filter data
    print(f"Cleaning and filtering data for {sentence_count} sentences...")
    df['step2_random_sent_num'] = pd.to_numeric(df['step2_random_sent_num'], errors='coerce')
    df['step3_output_int'] = pd.to_numeric(df['step3_output_int'], errors='coerce')
    
    # Filter to valid data - adjust range based on sentence count
    valid_range = list(range(1, sentence_count + 1))
    df = df[df['step2_random_sent_num'].isin(valid_range)]
    
    # Define the distribution computation function
    def compute_distribution(series, label):
        valid_series = series[series.isin(valid_range)]
        if valid_series.empty:
            return pd.DataFrame(columns=['step_num', 'percentage', 'source'])
            
        pct = (valid_series.value_counts(normalize=True) * 100).reset_index()
        pct.columns = ['step_num', 'percentage']
        pct['step_num'] = pct['step_num'].astype(int)
        pct['source'] = label
        
        # Ensure all step_num values from 1 to sentence_count exist
        # This fixes the discontinuity issue
        full_range_df = pd.DataFrame({'step_num': valid_range})
        pct = pd.merge(full_range_df, pct, on='step_num', how='left')
        # Fix pandas FutureWarning by avoiding inplace operations on DataFrame slices
        pct = pct.assign(
            source=pct['source'].fillna(label),
            percentage=pct['percentage'].fillna(0)
        )
        pct = pct.sort_values('step_num')
        
        return pct

    # Calculate percentage distributions
    step2_dist = compute_distribution(df['step2_random_sent_num'], '# of actual revised sentence')
    step3_dist = compute_distribution(df['step3_output_int'], '# of "strange" sentence')
    
    # Calculate accuracy per step2_random_sent_num
    accuracy_data = df.copy()
    accuracy_data['correct'] = accuracy_data['step2_random_sent_num'] == accuracy_data['step3_output_int']
    
    # Create a dataframe with all sentence numbers to ensure no gaps
    full_range_df = pd.DataFrame({'step2_random_sent_num': valid_range})
    
    # Calculate accuracy, handling the case where some sentence numbers might not appear
    accuracy_grouped = accuracy_data.groupby('step2_random_sent_num')['correct'].mean().reset_index()
    accuracy_df = pd.merge(full_range_df, accuracy_grouped, on='step2_random_sent_num', how='left')
    # Fix pandas FutureWarning by avoiding inplace operations on DataFrame slices
    accuracy_df = accuracy_df.assign(correct=accuracy_df['correct'].fillna(0))
    accuracy_df['percentage'] = accuracy_df['correct'] * 100
    accuracy_df['step_num'] = accuracy_df['step2_random_sent_num'].astype(int)
    accuracy_df['source'] = 'accuracy'
    accuracy_df = accuracy_df[['step_num', 'percentage', 'source']]
    accuracy_df = accuracy_df.sort_values('step_num')  # Ensure sorted by step_num

    # Get list of models from data
    models = list(set(df['Model_1'].unique()) | set(df['Model_2'].unique()))
    models = [m for m in models if isinstance(m, str)]
    
    # Model-specific distributions for step3_output_int
    model1_dists = []
    model2_dists = []
    
    print(f"Computing model-specific distributions for {sentence_count} sentences...")
    for model in models:
        subset = df[df['Model_1'] == model]
        if not subset.empty and subset['step3_output_int'].notna().any():
            valid_subset = subset[subset['step3_output_int'].isin(valid_range)]
            if not valid_subset.empty:
                dist = compute_distribution(valid_subset['step3_output_int'], f'Model_1:{model}')
                model1_dists.append(dist)

    for model in models:
        subset = df[df['Model_2'] == model]
        if not subset.empty and subset['step3_output_int'].notna().any():
            valid_subset = subset[subset['step3_output_int'].isin(valid_range)]
            if not valid_subset.empty:
                dist = compute_distribution(valid_subset['step3_output_int'], f'Model_2:{model}')
                model2_dists.append(dist)

    # Store distributions in dataframes
    model1_df = pd.concat(model1_dists, ignore_index=True) if model1_dists else pd.DataFrame()
    model2_df = pd.concat(model2_dists, ignore_index=True) if model2_dists else pd.DataFrame()

    print(f"Generating plots for {sentence_count} sentences...")

    # FIGURE 1: Combined standard distributions
    summary_df = pd.concat([
        step2_dist, 
        step3_dist,
        accuracy_df
    ]).reset_index(drop=True)
    
    # Filter out empty rows
    summary_df = summary_df[summary_df['step_num'].notna()]
    
    # Debug information
    print(f"\nData for Figure 1 ({sentence_count} sentences):")
    for source in summary_df['source'].unique():
        source_data = summary_df[summary_df['source'] == source].sort_values('step_num')
        print(f"\n{source}:")
        print(source_data[['step_num', 'percentage']].to_string(index=False))
    
    # If we have data, create the plot
    if not summary_df.empty:
        plt.figure(figsize=(12, 8))
        
        # Set colors for consistent appearance
        colors = {
            '# of actual revised sentence': 'tab:blue',
            '# of "strange" sentence': 'tab:orange',
            'accuracy': 'tab:green'
        }
        
        # Plot each source as a separate line
        for source in summary_df['source'].unique():
            source_data = summary_df[summary_df['source'] == source].sort_values('step_num')
            plt.plot(source_data['step_num'], source_data['percentage'], 
                     marker='o', linewidth=2, label=source, color=colors.get(source, None))
        
        plt.xlabel('Sentence Number', fontsize=12)
        plt.ylabel('Percentage', fontsize=12)
        plt.title(f'Sentence Number Distribution & Accuracy ({sentence_count} Sentences)', fontsize=14)
        plt.xticks(valid_range)
        plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 
                  ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(loc='upper right')
        
        # Save the figure
        plot1_path = os.path.join(sentence_dir, f"figure1_step_number_summary_{sentence_count}.png")
        plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {plot1_path}")
    else:
        print(f"Warning: No data available for Figure 1 ({sentence_count} sentences)")
    
    # FIGURE 2: Model_1 Distributions
    if not model1_df.empty:
        plt.figure(figsize=(12, 8))
        
        # Group by model and ensure each has a complete set of points
        model_groups = model1_df.groupby('source')
        for name, group in model_groups:
            group_sorted = group.sort_values('step_num')
            plt.plot(group_sorted['step_num'], group_sorted['percentage'], 
                     marker='o', linewidth=2, label=name)
        
        plt.xlabel('Sentence Number', fontsize=12)
        plt.ylabel('Percentage', fontsize=12)
        plt.title(f'Figure 2: Step 3 Output Distribution by Model 1 ({sentence_count} Sentences)', fontsize=14)
        plt.xticks(valid_range)
        plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 
                  ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(loc='best', bbox_to_anchor=(1.02, 1))
        
        # Save the figure
        plot2_path = os.path.join(sentence_dir, f"figure2_model1_distribution_{sentence_count}.png")
        plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {plot2_path}")
    else:
        print(f"Warning: No data available for Figure 2 ({sentence_count} sentences)")
    
    # FIGURE 3: Model_2 Distributions
    if not model2_df.empty:
        plt.figure(figsize=(12, 8))
        
        # Group by model and ensure each has a complete set of points
        model_groups = model2_df.groupby('source')
        for name, group in model_groups:
            group_sorted = group.sort_values('step_num')
            plt.plot(group_sorted['step_num'], group_sorted['percentage'], 
                     marker='o', linewidth=2, label=name)
        
        plt.xlabel('Sentence Number', fontsize=12)
        plt.ylabel('Percentage', fontsize=12)
        plt.title(f'Figure 3: Step 3 Output Distribution by Model 2 ({sentence_count} Sentences)', fontsize=14)
        plt.xticks(valid_range)
        plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 
                  ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(loc='best', bbox_to_anchor=(1.02, 1))
        
        # Save the figure
        plot3_path = os.path.join(sentence_dir, f"figure3_model2_distribution_{sentence_count}.png")
        plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {plot3_path}")
    else:
        print(f"Warning: No data available for Figure 3 ({sentence_count} sentences)")

    # Save the model pairs accuracy as CSV only
    model_pairs_df = (
        accuracy_data.groupby(['Model_1', 'Model_2'])['correct']
        .mean()
        .reset_index()
        .rename(columns={'correct': 'accuracy'})
    )
    model_pairs_df['accuracy'] *= 100
    
    if not model_pairs_df.empty:
        # Save as CSV
        model_pairs_path = os.path.join(sentence_dir, f"model_pair_accuracy_{sentence_count}.csv")
        model_pairs_df.to_csv(model_pairs_path, index=False)
        print(f"Saved: {model_pairs_path}")

    return df

def analyze_sample_sizes(data_dir):
    """Analyze the sample sizes across different sentence counts"""
    print("Analyzing sample sizes across sentence counts...")
    
    # Sentence counts to look for
    sentence_counts = [2, 3, 5, 7, 10, 13, 15, 18, 20]
    
    # Store the sample size data
    sample_size_data = []
    
    for sentence_count in sentence_counts:
        # Find all files for this sentence count
        matching_files = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(f"_{sentence_count}_sentences.csv"):
                    matching_files.append(os.path.join(root, file))
        
        if matching_files:
            # Load and count samples
            total_samples = 0
            for file_path in matching_files:
                try:
                    df = pd.read_csv(file_path, sep="|", engine="python")
                    total_samples += len(df)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
            
            sample_size_data.append({
                'sentence_count': sentence_count,
                'num_files': len(matching_files),
                'total_samples': total_samples
            })
    
    if sample_size_data:
        return pd.DataFrame(sample_size_data)
    else:
        return pd.DataFrame()

def main():
    # Paths
    OUTPUT_PATH = "./output/"
    FIGURE_PATH = os.path.join(OUTPUT_PATH, "figure/effect_of_length_of_sentences/")
    os.makedirs(FIGURE_PATH, exist_ok=True)
    
    # Data directory for sentence count results
    SENTENCE_COUNT_DIR = "./data/sentence_count_result/"
    
    # New Analysis: Sample Size Analysis
    print("\n===== Analyzing Sample Sizes =====")
    sample_size_df = analyze_sample_sizes(SENTENCE_COUNT_DIR)
    
    if not sample_size_df.empty:
        print(f"Sample size analysis complete. Found data for {len(sample_size_df)} sentence counts.")
    else:
        print("No sample size data found for analysis.")
    
    # Find all result files with different sentence counts
    print("\n===== Analyzing Results =====")
    print("Finding result files for different sentence counts...")
    
    # Sentence counts to look for
    sentence_counts = [2, 3, 5, 7, 10, 13, 15, 18, 20]
    
    # Process each sentence count
    for sentence_count in tqdm(sentence_counts, desc="Processing sentence counts"):
        process_sentence_count_data(sentence_count, SENTENCE_COUNT_DIR, FIGURE_PATH)

    print(f"Analysis complete. All plots saved to {FIGURE_PATH}")

if __name__ == "__main__":
    main()