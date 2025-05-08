import os
import pandas as pd
import numpy as np
from plotnine import *
from tqdm import tqdm
from scipy import stats

def main():
    # Configuration
    INPUT_DATA_PATH = "./data/step3/output_processed/"
    OUTPUT_PATH = "./output/"
    FIGURE_PATH = f"{OUTPUT_PATH}figure/randomization_check/"
    TABLE_PATH = f"{OUTPUT_PATH}table/"
    OUTPUT_FILE_TEMPLATE = f"{INPUT_DATA_PATH}mirror_test_results_{{}}_{{}}_{{}}_sentences.csv"
    
    # Create output directories if they don't exist
    os.makedirs(FIGURE_PATH, exist_ok=True)
    os.makedirs(TABLE_PATH, exist_ok=True)
    
    # Define models
    MODELS = ["ChatGPT", "Claude", "Grok", "Gemini", "Llama", "Deepseek"]
    
    # The sentence count is always 5 in this dataset
    SENTENCE_COUNT = 5
    
    # Track all data
    all_data = []
    
    # Process each model pair
    print("\n===== Analyzing Results for Randomization Check =====")
    print(f"Analyzing files for sentence count: {SENTENCE_COUNT}")
    
    total_pairs = len(MODELS) * len(MODELS)
    pbar = tqdm(total=total_pairs, desc="Processing model pairs")
    
    for model1 in MODELS:
        for model2 in MODELS:
            # Update the file path to match the actual structure
            file_path = f"{INPUT_DATA_PATH}mirror_test_results_{model1}_{model2}.csv"
            pbar.update(1)
            
            # Skip if file doesn't exist
            if not os.path.exists(file_path):
                continue
                
            try:
                # Read the CSV with pipe separator
                df = pd.read_csv(file_path, sep="|", engine="python")
                
                # Ensure numeric values
                df['step2_random_sent_num'] = pd.to_numeric(df['step2_random_sent_num'], errors='coerce')
                
                # Add metadata
                df['model1'] = model1
                df['model2'] = model2
                df['sentence_count'] = SENTENCE_COUNT
                
                # Add to our collection
                all_data.append(df)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    pbar.close()
    
    # Combine all data
    if not all_data:
        print("No data files found or processed successfully.")
        return
        
    print("Combining data from all model pairs...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Calculate distribution of actual modified sentences
    print("Calculating distribution of modified sentences...")
    
    # Get distribution for sentence count 5
    max_sentence = SENTENCE_COUNT
    
    # Filter to only relevant sentence numbers (1 to max_sentence)
    valid_subset = combined_df[
        combined_df['step2_random_sent_num'].between(1, max_sentence)
    ]
    
    if valid_subset.empty:
        print("No valid data found after filtering.")
        return
        
    # Distribution of actual modified sentences
    actual_dist = (
        valid_subset['step2_random_sent_num'].value_counts(normalize=True) * 100
    ).reset_index()
    actual_dist.columns = ['step_num', 'percentage']
    actual_dist['step_num'] = actual_dist['step_num'].astype(int)
    actual_dist['sentence_count'] = SENTENCE_COUNT
    actual_dist['source'] = 'Actual Modified Sentence'
    
    # Create the plot
    print("Generating randomization check visualization...")
    
    # For sentence count 5, we only have positions 1-5
    plot = (
        ggplot(actual_dist, aes(x='step_num', y='percentage'))
        + geom_bar(stat='identity', fill='steelblue', alpha=0.8)
        + geom_hline(yintercept=20, linetype='dashed', color='red', size=1)  # Expected uniform distribution
        + scale_x_continuous(breaks=[1, 2, 3, 4, 5], limits=[0.5, 5.5])
        + scale_y_continuous(
            limits=[0, 40],
            breaks=[0, 5, 10, 15, 20, 25, 30, 35, 40],
            labels=lambda x: [f"{v}%" for v in x]
        )
        + theme_minimal()
        + theme(
            figure_size=(8, 6),
            panel_grid_minor=element_blank(),
            panel_grid_major_y=element_line(color='lightgray'),
            panel_grid_major_x=element_blank(),
            text=element_text(size=12),
            axis_text=element_text(size=11),
            plot_title=element_text(size=16, face="bold"),
            axis_title=element_text(size=13)
        )
        + labs(
            title='Randomization Check: Distribution of Modified Sentences',
            x='Sentence Position',
            y='Percentage of Times Modified',
        )
    )
    
    # Save plot
    plot_path = os.path.join(FIGURE_PATH, "randomization_check_sentence_5.png")
    plot.save(plot_path, dpi=600, width=8, height=6)
    print(f"Saved randomization check plot to: {plot_path}")
    
    # Statistical test for uniformity
    print("\n===== Statistical Test for Randomization =====")
    
    # Expected uniform distribution (20% for each position in 5-sentence stories)
    expected_percentage = 100.0 / SENTENCE_COUNT
    observed_counts = actual_dist['percentage'].values
    
    # Chi-square test for uniformity
    chi_stat, p_value = stats.chisquare(observed_counts, f_exp=[expected_percentage] * len(observed_counts))
    
    # Create results table
    chi_square_results = {
        'sentence_count': SENTENCE_COUNT,
        'chi_square': chi_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'interpretation': 'Significantly different from uniform' if p_value < 0.05 else 'Not significantly different from uniform'
    }
    
    # Save results
    chi_square_df = pd.DataFrame([chi_square_results])
    chi_square_path = os.path.join(TABLE_PATH, "randomization_check_chi_square_results.csv")
    chi_square_df.to_csv(chi_square_path, index=False)
    print(f"Saved chi-square test results to: {chi_square_path}")
    
    # Print summary
    print(f"\nChi-square test results for sentence count {SENTENCE_COUNT}:")
    print(f"  chi² = {chi_stat:.2f}")
    print(f"  p-value = {p_value:.4f}")
    print(f"  Result: {chi_square_results['interpretation']}")
    
    # Print detailed distribution
    print("\nDetailed distribution:")
    for _, row in actual_dist.iterrows():
        print(f"  Sentence {int(row['step_num'])}: {row['percentage']:.1f}%")
    
    # Add analysis of valid responses percentage and recognition accuracy per model pair
    print("\n===== Analyzing Valid Responses Percentage and Recognition Accuracy per Model Pair =====")
    
    # Calculate valid responses and recognition accuracy for each model pair
    model_pair_stats = []
    
    for model1 in MODELS:
        for model2 in MODELS:
            # Get data for this model pair
            model_pair_data = combined_df[(combined_df['model1'] == model1) & (combined_df['model2'] == model2)]
            
            # Calculate percentage of valid responses (those with sentence numbers between 1 and max_sentence)
            total_responses = len(model_pair_data)
            
            if total_responses == 0:
                # No data for this pair
                model_pair_stats.append({
                    'model1': model1,
                    'model2': model2,
                    'model_pair': f"{model1}-{model2}",
                    'total_responses': 0,
                    'valid_responses': 0,
                    'valid_percentage': float('nan'),  # Use NaN for missing data
                    'correct_recognitions': 0,
                    'recognition_accuracy': float('nan')  # Use NaN for missing data
                })
                continue
                
            # Valid responses (those with sentence numbers between 1 and max_sentence)
            valid_responses = len(model_pair_data[model_pair_data['step2_random_sent_num'].between(1, max_sentence)])
            valid_percentage = (valid_responses / total_responses) * 100
            
            # Recognition accuracy (correct identifications of the modified sentence)
            # Check if step3_output_int and step2_random_sent_num match
            if 'step3_output_int' in model_pair_data.columns:
                # Create a copy first to avoid the SettingWithCopyWarning
                data_for_analysis = model_pair_data.copy()
                
                # Convert to numeric, ignoring errors
                data_for_analysis['step3_output_int'] = pd.to_numeric(data_for_analysis['step3_output_int'], errors='coerce')
                
                # Only consider valid responses for recognition accuracy
                valid_subset = data_for_analysis[data_for_analysis['step2_random_sent_num'].between(1, max_sentence)]
                
                # Check if is_correct column is available
                if 'is_correct' in model_pair_data.columns:
                    # Use is_correct column if available
                    correct_recognitions = len(valid_subset[valid_subset['is_correct'] == True])
                else:
                    # Else count matches between predicted and actual sentence numbers
                    correct_recognitions = len(valid_subset[valid_subset['step3_output_int'] == valid_subset['step2_random_sent_num']])
                
                recognition_accuracy = (correct_recognitions / valid_responses) * 100 if valid_responses > 0 else 0
            else:
                correct_recognitions = 0
                recognition_accuracy = 0
            
            model_pair_stats.append({
                'model1': model1,
                'model2': model2,
                'model_pair': f"{model1}-{model2}",
                'total_responses': total_responses,
                'valid_responses': valid_responses,
                'valid_percentage': valid_percentage,
                'correct_recognitions': correct_recognitions,
                'recognition_accuracy': recognition_accuracy
            })
    
    # Create dataframe from stats
    model_pair_df = pd.DataFrame(model_pair_stats)
    
    if not model_pair_df.empty:
        # Create plot for valid responses percentage
        print("Generating valid responses percentage visualization...")
        
        # Clone the DataFrame and replace NaN values with specific text for display
        plot_df = model_pair_df.copy()
        
        # Create label column that handles NaN values
        plot_df['label_text'] = plot_df['valid_percentage'].apply(
            lambda x: 'No Data' if pd.isna(x) else f'{x:.0f}%'
        )
        
        # For plotting purposes, replace NaN with -1 (we'll use a separate color for these)
        plot_df['valid_percentage_plot'] = plot_df['valid_percentage'].fillna(-1)
        
        # Plot valid response percentages
        valid_plot = (
            ggplot(plot_df, aes(x='model1', y='model2', fill='valid_percentage_plot'))
            + geom_tile(aes(width=0.95, height=0.95))
            + scale_fill_gradient2(
                low='lightcoral', mid='khaki', high='lightgreen',
                midpoint=75, limits=[-1, 100],
                name='Valid\nResponses (%)',
                na_value='lightgray'
            )
            + geom_text(aes(label='label_text'), size=8)
            + theme_minimal()
            + theme(
                figure_size=(10, 8),
                panel_grid=element_blank(),
                text=element_text(size=12),
                axis_text_x=element_text(angle=45, hjust=1),
                plot_title=element_text(size=16, face="bold"),
                axis_title=element_text(size=13)
            )
            + labs(
                title='Valid Responses Percentage by Model Pair',
                x='Model 1 (Story Generator)',
                y='Model 2 (Story Modifier)'
            )
        )
        
        # Save plot
        valid_plot_path = os.path.join(FIGURE_PATH, "valid_responses_percentage.png")
        valid_plot.save(valid_plot_path, dpi=600, width=10, height=8)
        print(f"Saved valid responses percentage plot to: {valid_plot_path}")
        
        # Create a similar plot for recognition accuracy
        # Clone the DataFrame and replace NaN values with specific text for display
        recog_df = model_pair_df.copy()
        
        # Create label column that handles NaN values
        recog_df['accuracy_text'] = recog_df['recognition_accuracy'].apply(
            lambda x: 'No Data' if pd.isna(x) else f'{x:.1f}%'
        )
        
        # For plotting purposes, replace NaN with -1 (we'll use a separate color for these)
        recog_df['recognition_accuracy_plot'] = recog_df['recognition_accuracy'].fillna(-1)
        
        # Plot recognition accuracy
        # Calculate expected random guess accuracy (1/number of sentences)
        random_guess_accuracy = 100.0 / SENTENCE_COUNT
        
        # Add a note about random chance to the dataframe for the plot title
        random_chance_note = f"(Random Chance: {random_guess_accuracy:.1f}%)"
        
        recognition_plot = (
            ggplot(recog_df, aes(x='model1', y='model2', fill='recognition_accuracy_plot'))
            + geom_tile(aes(width=0.95, height=0.95))
            + scale_fill_gradient2(
                low='#FFCCCC', mid='#FFFFCC', high='#CCFFCC',
                midpoint=30, limits=[-1, 60],
                name='Recognition\nAccuracy (%)',
                na_value='lightgray'
            )
            + geom_text(aes(label='accuracy_text'), size=8)
            + theme_minimal()
            + theme(
                figure_size=(10, 8),
                panel_grid=element_blank(),
                text=element_text(size=12),
                axis_text_x=element_text(angle=45, hjust=1),
                plot_title=element_text(size=16, face="bold"),
                axis_title=element_text(size=13)
            )
            + labs(
                title=f'Recognition Accuracy by Model Pair {random_chance_note}',
                x='Model 1 (Story Generator)',
                y='Model 2 (Story Modifier)'
            )
        )
        
        # Save recognition accuracy plot
        recognition_plot_path = os.path.join(FIGURE_PATH, "recognition_accuracy.png")
        recognition_plot.save(recognition_plot_path, dpi=600, width=10, height=8)
        print(f"Saved recognition accuracy plot to: {recognition_plot_path}")
        
        # Save stats as CSV
        stats_path = os.path.join(TABLE_PATH, "valid_responses_stats.csv")
        model_pair_df.to_csv(stats_path, index=False)
        print(f"Saved valid responses stats to: {stats_path}")

if __name__ == "__main__":
    main()