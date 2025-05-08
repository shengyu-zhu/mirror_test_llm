import os
import pandas as pd
import numpy as np
from plotnine import *
import argparse
from tqdm import tqdm

def generate_latex_table(df):
    """
    Generate a LaTeX table from a pandas DataFrame containing overall metrics.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with columns: sentence_count, accuracy, precision, recall, f1_score, sample_size
    
    Returns:
    str: LaTeX table code
    """
    # Table header with booktabs
    latex_table = """% Requires \\usepackage{booktabs} in preamble
\\begin{table}[htbp]
  \\centering
  \\caption{Overall Performance Metrics by Story Length}
  \\begin{tabular}{rrrrrc}
    \\toprule
    \\multicolumn{1}{c}{\\textbf{\\# of Sentences}} & 
    \\multicolumn{1}{c}{\\textbf{Accuracy (\\%)}} & 
    \\multicolumn{1}{c}{\\textbf{Precision (\\%)}} & 
    \\multicolumn{1}{c}{\\textbf{Recall (\\%)}} & 
    \\multicolumn{1}{c}{\\textbf{F1 Score}} & 
    \\multicolumn{1}{c}{\\textbf{n}} \\\\
    \\midrule
"""
    
    # Sort by sentence count
    df_sorted = df.sort_values('sentence_count')
    
    # Table rows with better formatting
    for _, row in df_sorted.iterrows():
        latex_table += f"    {int(row['sentence_count'])} & {row['accuracy']:.1f} & {row['precision']:.1f} & {row['recall']:.1f} & {row['f1_score']:.1f} & {int(row['sample_size']):,} \\\\\n"
    
    # Table footer
    latex_table += """    \\bottomrule
  \\end{tabular}
  \\label{tab:overall-metrics}
\\end{table}
"""
    
    return latex_table

def analyze_sample_sizes(sentence_count_dir):
    """
    Analyze and compare initial sample sizes and filtered sample sizes.
    
    Parameters:
    sentence_count_dir (str): Directory containing the experiment result files
    
    Returns:
    pd.DataFrame: DataFrame with sample size analysis
    """
    print("Analyzing initial and filtered sample sizes...")
    
    # Initialize data structure to hold the results
    sample_size_data = []
    sentence_counts = [2, 3, 5, 7, 10, 13, 15, 18, 20]
    
    for sentence_count in sentence_counts:
        # Look for step1 files that contain initial generation data
        step1_files = [f for f in os.listdir(sentence_count_dir) 
                       if f.startswith("step1_generate_stories_") and f"_{sentence_count}_sentences_" in f]
        
        # Look for result files that contain filtered data
        result_files = [f for f in os.listdir(sentence_count_dir) 
                        if f.startswith("mirror_test_results_") and f"_{sentence_count}_sentences.csv" in f]
        
        for step1_file in step1_files:
            try:
                # Extract model name from filename
                model_name = step1_file.split("_")[3]
                
                # Read step1 file to get initial sample size
                step1_path = os.path.join(sentence_count_dir, step1_file)
                step1_df = pd.read_csv(step1_path, sep='|')
                
                # Get initial sample size (total rows)
                initial_size = len(step1_df)
                
                # Get number of correctly generated stories (with exactly the requested number of sentences)
                correct_size = step1_df["step1_m1_output_sent_count"].value_counts().get(sentence_count, 0)
                
                # Try to find corresponding result file
                matching_result_files = [f for f in result_files if model_name in f]
                final_size = 0
                
                if matching_result_files:
                    # Read result file to get final sample size
                    result_path = os.path.join(sentence_count_dir, matching_result_files[0])
                    result_df = pd.read_csv(result_path, sep='|')
                    final_size = len(result_df)
                
                # Add to data collection
                sample_size_data.append({
                    'sentence_count': sentence_count,
                    'model': model_name,
                    'initial_size': initial_size,
                    'correct_size': correct_size, 
                    'final_size': final_size,
                    'correct_percentage': (correct_size / initial_size * 100) if initial_size > 0 else 0,
                    'final_percentage': (final_size / initial_size * 100) if initial_size > 0 else 0
                })
                
            except Exception as e:
                print(f"Error processing {step1_file}: {e}")
    
    # Create DataFrame
    if sample_size_data:
        return pd.DataFrame(sample_size_data)
    else:
        return pd.DataFrame(columns=['sentence_count', 'model', 'initial_size', 'correct_size', 
                                     'final_size', 'correct_percentage', 'final_percentage'])

def plot_sample_size_analysis(sample_size_df, figure_path):
    """
    Create plots for sample size analysis.
    
    Parameters:
    sample_size_df (pd.DataFrame): DataFrame with sample size analysis
    figure_path (str): Directory to save plot files
    """
    if sample_size_df.empty:
        print("No sample size data to plot.")
        return
    
    # 1. Plot absolute sample sizes by sentence count
    plot_abs = (
        ggplot(sample_size_df, aes(x='sentence_count', y='initial_size', fill='model'))
        + geom_bar(stat='identity', position='dodge')
        + geom_text(aes(label='initial_size'), position=position_dodge(width=0.9), va='bottom', format_string='{:.0f}', size=8)
        + scale_x_continuous(breaks=sorted(sample_size_df['sentence_count'].unique()))
        + theme_minimal()
        + theme(
            legend_position='bottom',
            figure_size=(1.618 * 5, 5),
            panel_grid_minor=element_blank()
        )
        + labs(
            title='Initial Sample Size by Story Length',
            x='Number of Sentences in Story',
            y='Number of Samples',
            fill='Model'
        )
    )
    
    plot_abs_path = os.path.join(figure_path, "initial_sample_size_by_sentence_count.png")
    plot_abs.save(plot_abs_path, dpi=600)
    print(f"Saved initial sample size plot to: {plot_abs_path}")
    
    # 2. Check if correct_percentage and final_percentage are always the same
    all_equal = all(
        abs(row['correct_percentage'] - row['final_percentage']) < 0.01  # Using small epsilon for float comparison
        for _, row in sample_size_df.iterrows()
    )
    
    # Print whether the percentages are the same
    if all_equal:
        print("\n*** ANALYSIS NOTE: 'Correct Sentence Count' and 'Final Dataset' percentages are identical ***")
        print("Using simplified plot with only final percentages.")
    else:
        print("\n*** ANALYSIS NOTE: 'Correct Sentence Count' and 'Final Dataset' percentages differ ***")
        print("Some samples were filtered out between these stages.")
    
    # The sample retention percentage will be plotted together with accuracy in the main function
    # No need to generate a separate plot here anymore
    
    # 3. Create a table with the sample size information
    table_df = sample_size_df.copy()
    # Add percentage symbols
    table_df['correct_percentage'] = table_df['correct_percentage'].apply(lambda x: f"{x:.1f}%")
    table_df['final_percentage'] = table_df['final_percentage'].apply(lambda x: f"{x:.1f}%")
    
    # Reorder columns for better readability
    table_df = table_df[['sentence_count', 'model', 'initial_size', 'correct_size', 
                        'correct_percentage', 'final_size', 'final_percentage']]
    
    # Save table to CSV
    table_path = os.path.join(figure_path, "sample_size_analysis.csv")
    table_df.to_csv(table_path, index=False)
    print(f"Saved sample size analysis table to: {table_path}")
    
    # Generate LaTeX table
    latex_table = """% Requires \\usepackage{booktabs} in preamble
\\begin{table}[htbp]
  \\centering
  \\caption{Sample Size Analysis by Story Length}
  \\begin{tabular}{ccrrrrc}
    \\toprule
    \\multicolumn{1}{c}{\\textbf{\\# of Sentences}} & 
    \\multicolumn{1}{c}{\\textbf{Model}} &
    \\multicolumn{1}{c}{\\textbf{Initial Size}} & 
    \\multicolumn{1}{c}{\\textbf{Correct Count}} & 
    \\multicolumn{1}{c}{\\textbf{Correct \\%}} & 
    \\multicolumn{1}{c}{\\textbf{Final Size}} &
    \\multicolumn{1}{c}{\\textbf{Final \\%}} \\\\
    \\midrule
"""
    
    # Sort by sentence count and model
    table_df_sorted = table_df.sort_values(['sentence_count', 'model'])
    
    # Table rows
    for _, row in table_df_sorted.iterrows():
        latex_table += f"    {int(row['sentence_count'])} & {row['model']} & {int(row['initial_size']):,} & {int(row['correct_size']):,} & {row['correct_percentage']} & {int(row['final_size']):,} & {row['final_percentage']} \\\\\n"
    
    # Table footer
    latex_table += """    \\bottomrule
  \\end{tabular}
  \\label{tab:sample-size-analysis}
\\end{table}
"""
    
    # Save LaTeX table
    latex_path = os.path.join(figure_path, "sample_size_analysis_latex_table.tex")
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"Saved LaTeX table to: {latex_path}")
    
    
    # 3. Create a table with the sample size information
    table_df = sample_size_df.copy()
    # Add percentage symbols
    table_df['correct_percentage'] = table_df['correct_percentage'].apply(lambda x: f"{x:.1f}%")
    table_df['final_percentage'] = table_df['final_percentage'].apply(lambda x: f"{x:.1f}%")
    
    # Reorder columns for better readability
    table_df = table_df[['sentence_count', 'model', 'initial_size', 'correct_size', 
                        'correct_percentage', 'final_size', 'final_percentage']]
    
    # Save table to CSV
    table_path = os.path.join(figure_path, "sample_size_analysis.csv")
    table_df.to_csv(table_path, index=False)
    print(f"Saved sample size analysis table to: {table_path}")
    
    # Generate LaTeX table
    latex_table = """% Requires \\usepackage{booktabs} in preamble
\\begin{table}[htbp]
  \\centering
  \\caption{Sample Size Analysis by Story Length}
  \\begin{tabular}{ccrrrrc}
    \\toprule
    \\multicolumn{1}{c}{\\textbf{\\# of Sentences}} & 
    \\multicolumn{1}{c}{\\textbf{Model}} &
    \\multicolumn{1}{c}{\\textbf{Initial Size}} & 
    \\multicolumn{1}{c}{\\textbf{Correct Count}} & 
    \\multicolumn{1}{c}{\\textbf{Correct \\%}} & 
    \\multicolumn{1}{c}{\\textbf{Final Size}} &
    \\multicolumn{1}{c}{\\textbf{Final \\%}} \\\\
    \\midrule
"""
    
    # Sort by sentence count and model
    table_df_sorted = table_df.sort_values(['sentence_count', 'model'])
    
    # Table rows
    for _, row in table_df_sorted.iterrows():
        latex_table += f"    {int(row['sentence_count'])} & {row['model']} & {int(row['initial_size']):,} & {int(row['correct_size']):,} & {row['correct_percentage']} & {int(row['final_size']):,} & {row['final_percentage']} \\\\\n"
    
    # Table footer
    latex_table += """    \\bottomrule
  \\end{tabular}
  \\label{tab:sample-size-analysis}
\\end{table}
"""
    
    # Save LaTeX table
    latex_path = os.path.join(figure_path, "sample_size_analysis_latex_table.tex")
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"Saved LaTeX table to: {latex_path}")

def plot_accuracy_above_random(distribution_df, sentence_counts, FIGURE_PATH):
    """
    Create a plot showing the difference between observed accuracy and random guess accuracy.
    
    Parameters:
    distribution_df (pd.DataFrame): DataFrame with distribution data including 'Accuracy' source
    sentence_counts (list): List of sentence counts to analyze
    FIGURE_PATH (str): Directory to save plot files
    """

    print("Generating accuracy above random chance plot...")
    
    # Filter only the accuracy data - create an explicit copy to avoid SettingWithCopyWarning
    accuracy_data = distribution_df[distribution_df['source'] == 'Accuracy'].copy()
    
    if accuracy_data.empty:
        print("No accuracy data available for plotting.")
        return
    
    # Calculate random guess accuracy for each sentence count
    accuracy_data.loc[:, 'random_accuracy'] = 100 / accuracy_data['sentence_count']
    
    # Calculate difference from random
    accuracy_data.loc[:, 'above_random'] = accuracy_data['percentage'] - accuracy_data['random_accuracy']
    
    # Create plot
    plot = (
        ggplot(accuracy_data, aes(x='step_num', y='above_random', color='factor(sentence_count)', group='factor(sentence_count)'))
        + geom_line(size=1.2)
        + geom_point(size=3)
        + geom_hline(yintercept=0, linetype='dashed', color='darkgray', size=0.8)
        + scale_x_continuous(breaks=[1, 3, 5, 7, 10, 13, 15, 18, 20], limits=[0.5, 20.5])
        + scale_y_continuous(breaks=[-20, 0, 20, 40, 60], limits=[-25, 65])
        + scale_color_brewer(type='qual', palette='Dark2')
        + theme_minimal()
        + theme(
            legend_position='bottom', 
            figure_size=(1.618 * 3, 3),
            panel_grid_minor=element_blank(),
            panel_grid_major_y=element_line(color='lightgray'),
            panel_grid_major_x=element_blank(),
        )
        + labs(
            title='Accuracy Above Random Chance',
            x='Sentence Number',
            y='% Above Random Chance',
            color='Sentence Count'
        )
    )
    
    # Add annotations for random chance accuracy per sentence count
    annotations = []
    for sc in sorted(accuracy_data['sentence_count'].unique()):
        random_acc = 100 / sc
        annotations.append({
            'sentence_count': sc,
            'text': f"{sc} sentences: random = {random_acc:.1f}%"
        })
    
    # # Create a list of colors for annotations
    # # Fix: Use a predefined list of colors instead of trying to access the palette directly
    # annotation_colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666']
    
    # # Add annotation texts to the right side of the plot
    # for i, row in enumerate(annotations):
    #     plot = plot + annotate(
    #         'text', 
    #         x=15, 
    #         y=(len(annotations) - i) * 5 - 5,  # Position from top to bottom
    #         label=row['text'], 
    #         ha='left', 
    #         size=8,
    #         color=annotation_colors[i % len(annotation_colors)]  # Use the color list safely
    #     )
    
    # Save plot
    plot_path = os.path.join(FIGURE_PATH, "accuracy_above_random_by_position.png")
    plot.save(plot_path, dpi=600)
    print(f"Saved accuracy above random plot to: {plot_path}")

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
    
    # New Analysis: Sample Size Analysis
    print("\n===== Analyzing Sample Sizes =====")
    sample_size_df = analyze_sample_sizes(SENTENCE_COUNT_DIR)
    
    if not sample_size_df.empty:
        # Plot and save sample size analysis
        plot_sample_size_analysis(sample_size_df, FIGURE_PATH)
    else:
        print("No sample size data found for analysis.")
    
    # Find all result files with different sentence counts
    print("\n===== Analyzing Results =====")
    print("Finding result files for different sentence counts...")
    
    # Track the data for each sentence count
    sentence_length_data = []
    
    # Sentence counts to analyze
    sentence_counts = [2, 3, 5, 7, 10, 13, 15, 18, 20]
    
    # Process each sentence count
    for sentence_count in tqdm(sentence_counts, desc="Processing sentence counts"):
        # Find files for this sentence count
        pattern = f"mirror_test_results_*_{sentence_count}_sentences.csv"
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
    
    # 2. Distribution of predicted "strange" sentences (step3_output_int)
    predicted_distributions = []
    
    # 3. Accuracy by sentence number for each sentence count
    accuracy_distributions = []
    
    # 4. Precision by sentence number for each sentence count
    precision_distributions = []
    
    # 5. Recall by sentence number for each sentence count
    recall_distributions = []
    
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
        
        # 2. Distribution of predicted "strange" sentences
        predicted_dist = (
            valid_subset['step3_output_int'].value_counts(normalize=True) * 100
        ).reset_index()
        predicted_dist.columns = ['step_num', 'percentage']
        predicted_dist['step_num'] = predicted_dist['step_num'].astype(int)
        predicted_dist['sentence_count'] = sentence_count
        predicted_dist['source'] = 'Predicted "Strange" Sentence'
        predicted_distributions.append(predicted_dist)
        
        # 3. Accuracy by sentence number
        valid_subset['correct'] = valid_subset['step2_random_sent_num'] == valid_subset['step3_output_int']
        accuracy_by_num = (
            valid_subset.groupby('step2_random_sent_num')['correct'].mean() * 100
        ).reset_index()
        accuracy_by_num.columns = ['step_num', 'percentage']
        accuracy_by_num['step_num'] = accuracy_by_num['step_num'].astype(int)
        accuracy_by_num['sentence_count'] = sentence_count
        accuracy_by_num['source'] = 'Accuracy'
        accuracy_distributions.append(accuracy_by_num)
        
        # 4. Calculate precision for each sentence number
        # Precision: Of all sentences the model predicted as modified at position X, 
        # what percentage were actually modified at position X?
        precision_by_num = []
        
        for step_num in range(1, max_sentence + 1):
            # Total predictions for this step number
            total_predicted = (valid_subset['step3_output_int'] == step_num).sum()
            
            if total_predicted > 0:
                # Correct predictions for this step number
                correct_predicted = ((valid_subset['step3_output_int'] == step_num) & 
                                    (valid_subset['step2_random_sent_num'] == step_num)).sum()
                
                # Calculate precision
                precision = (correct_predicted / total_predicted) * 100
            else:
                precision = 0
                
            precision_by_num.append({
                'step_num': step_num,
                'percentage': precision,
                'sentence_count': sentence_count,
                'source': 'Precision'
            })
        
        precision_df = pd.DataFrame(precision_by_num)
        precision_distributions.append(precision_df)
        
        # 5. Calculate recall for each sentence number
        # Recall: Of all sentences actually modified at position X, 
        # what percentage did the model correctly identify?
        recall_by_num = []
        
        for step_num in range(1, max_sentence + 1):
            # Total actual instances of this step number
            total_actual = (valid_subset['step2_random_sent_num'] == step_num).sum()
            
            if total_actual > 0:
                # Correct predictions for this step number
                correct_predicted = ((valid_subset['step3_output_int'] == step_num) & 
                                    (valid_subset['step2_random_sent_num'] == step_num)).sum()
                
                # Calculate recall
                recall = (correct_predicted / total_actual) * 100
            else:
                recall = 0
                
            recall_by_num.append({
                'step_num': step_num,
                'percentage': recall,
                'sentence_count': sentence_count,
                'source': 'Recall'
            })
        
        recall_df = pd.DataFrame(recall_by_num)
        recall_distributions.append(recall_df)
    
    # Combine all distributions
    all_dists = []
    
    if actual_distributions:
        actual_df = pd.concat(actual_distributions, ignore_index=True)
        all_dists.append(actual_df)
        
    if predicted_distributions:
        predicted_df = pd.concat(predicted_distributions, ignore_index=True)
        all_dists.append(predicted_df)
        
    if accuracy_distributions:
        accuracy_df = pd.concat(accuracy_distributions, ignore_index=True)
        all_dists.append(accuracy_df)
        
    if precision_distributions:
        precision_df = pd.concat(precision_distributions, ignore_index=True)
        all_dists.append(precision_df)
        
    if recall_distributions:
        recall_df = pd.concat(recall_distributions, ignore_index=True)
        all_dists.append(recall_df)
    
    if not all_dists:
        print("No valid distributions could be calculated.")
        return
        
    distribution_df = pd.concat(all_dists, ignore_index=True)

    # Call the function to plot accuracy above random
    plot_accuracy_above_random(distribution_df, sentence_counts, FIGURE_PATH)
    
    # Call the new function to plot predicted strange sentence above random
    plot_predicted_above_random(distribution_df, sentence_counts, FIGURE_PATH)

    print("Generating plots...")
    # Rest of the main function continues as before...

    print("Generating plots...")
    # Calculate overall metrics for each sentence count
    print("Calculating overall metrics by sentence count...")
    overall_metrics = []
    
    for sentence_count in sentence_counts:
        subset = combined_df[combined_df['sentence_count'] == sentence_count].copy()  # Create explicit copy
        
        if subset.empty:
            continue
            
        # Get max sentence number (up to 20 for extended range)
        max_sentence = min(20, sentence_count)
        
        # Filter to only relevant sentence numbers
        valid_subset = subset[
            subset['step2_random_sent_num'].between(1, max_sentence) & 
            subset['step3_output_int'].between(1, max_sentence)
        ].copy()  # Create explicit copy
        
        if valid_subset.empty:
            continue
            
        # Calculate overall accuracy - use proper pandas assignment
        valid_subset.loc[:, 'correct'] = valid_subset['step2_random_sent_num'] == valid_subset['step3_output_int']
        accuracy = valid_subset['correct'].mean() * 100
        
        # Set up confusion matrix for overall metrics
        confusion_matrix = np.zeros((max_sentence+1, max_sentence+1), dtype=int)
        
        # Fill the confusion matrix
        for _, row in valid_subset.iterrows():
            actual = int(row['step2_random_sent_num'])
            predicted = int(row['step3_output_int'])
            confusion_matrix[actual, predicted] += 1
        
        # Initialize counters for each sentence position
        precision_values = []
        recall_values = []
        
        # Calculate precision and recall for each sentence position
        for i in range(1, max_sentence+1):
            # True positives: correctly identified sentence i
            tp = confusion_matrix[i, i]
            
            # False positives: predicted sentence i but was wrong
            fp = confusion_matrix[:, i].sum() - tp
            
            # False negatives: actual sentence was i but predicted something else
            fn = confusion_matrix[i, :].sum() - tp
            
            # Calculate precision and recall for this position
            pos_precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
            pos_recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
            
            precision_values.append(pos_precision)
            recall_values.append(pos_recall)
        
        # Overall precision and recall (macro average)
        precision = sum(precision_values) / len(precision_values) if precision_values else 0
        recall = sum(recall_values) / len(recall_values) if recall_values else 0
        
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Add to overall metrics
        overall_metrics.append({
            'sentence_count': sentence_count,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'sample_size': len(valid_subset)
        })
    
    # Create overall metrics dataframe
    if overall_metrics:
        overall_df = pd.DataFrame(overall_metrics)
        
        # Save overall metrics to CSV
        overall_csv_path = os.path.join(FIGURE_PATH, "overall_metrics_by_sentence_count.csv")
        overall_df.to_csv(overall_csv_path, index=False)
        print(f"Saved overall metrics to: {overall_csv_path}")
        
        # Generate LaTeX table
        latex_table = generate_latex_table(overall_df)
        latex_path = os.path.join(FIGURE_PATH, "overall_metrics_latex_table.tex")
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        print(f"Saved LaTeX table to: {latex_path}")
        
 # Create a combined plot of sample retention and accuracy
        if not sample_size_df.empty:
            print("Generating combined accuracy and retention plot...")

            # Get only the accuracy metric from overall_df
            accuracy_df = overall_df[['sentence_count', 'accuracy', 'sample_size']].copy()
            accuracy_df['metric'] = 'Accuracy'
            accuracy_df['percentage'] = accuracy_df['accuracy']
            
            # Get sample retention percentages - aggregating across models if multiple exist
            retention_agg = sample_size_df.groupby('sentence_count').agg({
                'initial_size': 'sum',       # Sum initial sizes
                'final_size': 'sum'          # Sum final sizes
            }).reset_index()

            # Calculate correct overall percentage
            retention_agg['overall_percentage'] = (retention_agg['final_size'] / retention_agg['initial_size']) * 100

            # Create retention dataframe for plotting
            retention_df = pd.DataFrame({
                'sentence_count': retention_agg['sentence_count'],
                'metric': 'Stories Meeting Sentence Count Criteria',
                'percentage': retention_agg['overall_percentage'],  # Use the correct overall percentage
                'sample_size': None  # Placeholder for consistency
            })
            print(retention_df)
            
            # Create random chance dataframe
            random_chance_df = pd.DataFrame({
                'sentence_count': overall_df['sentence_count'],
                'metric': 'Random Guess Chance',
                'percentage': 100 / overall_df['sentence_count'],  # Random chance = 1/n
                'sample_size': None  # Placeholder for consistency
            })
            
            # Combine the dataframes for plotting
            combined_df = pd.concat([
                accuracy_df[['sentence_count', 'metric', 'percentage', 'sample_size']],
                retention_df[['sentence_count', 'metric', 'percentage', 'sample_size']],
                random_chance_df[['sentence_count', 'metric', 'percentage', 'sample_size']]
            ])
            
            # Make sure all sentence_count values are numeric
            combined_df['sentence_count'] = pd.to_numeric(combined_df['sentence_count'])
            
            # Get unique sentence counts and their corresponding sample sizes from accuracy_df
            unique_sentence_counts = accuracy_df['sentence_count'].tolist()
            sample_sizes = accuracy_df['sample_size'].tolist()
            
            try:
                # Create combined line plot with random chance line
                combined_plot = (
                    ggplot(combined_df, aes(x='sentence_count', y='percentage', color='metric', group='metric'))
                    + geom_line(size=1.2, alpha=0.7)
                    + geom_point(size=3.5, alpha=0.8)
                    + scale_x_continuous(breaks=sorted(combined_df['sentence_count'].unique()))
                    + scale_y_continuous(
                        limits=[0, 100],
                        breaks=[0, 20, 40, 60, 80, 100],
                        labels=lambda x: [f"{v}%" for v in x]
                    )
                    + scale_color_brewer(type='qual', palette='Set1')
                    + theme_minimal()
                    + theme(
                        legend_position='bottom', 
                        figure_size=(1.618 * 4, 4),  # Slightly wider to accommodate the additional line
                        panel_grid_minor=element_blank(),
                        panel_grid_major_y=element_line(color='lightgray'),
                        panel_grid_major_x=element_blank(),
                    )
                    + labs(
                        title='Accuracy, Sample Retention, and Random Chance by Story Length',
                        x='Number of Sentences in Story',
                        y='Percentage',
                        color='Metric'
                    )
                )
                
                # Add sample size annotations
                for i, (sc, ss) in enumerate(zip(unique_sentence_counts, sample_sizes)):
                    combined_plot = combined_plot + annotate('text', x=sc, y=-5, label=f"n={ss}", size=8)
                
                combined_plot_path = os.path.join(FIGURE_PATH, "accuracy_and_retention_by_sentence_count.png")
                combined_plot.save(combined_plot_path, dpi=600)
                print(f"Saved combined accuracy and retention plot to: {combined_plot_path}")
            except Exception as e:
                print(f"Error generating combined plot: {e}")
                # Print debugging information
                print("\nDebugging information:")
                print(f"Combined DataFrame shape: {combined_df.shape}")
                print(f"Combined DataFrame head:\n{combined_df.head()}")
                print(f"Unique sentence counts: {unique_sentence_counts}")
                print(f"Sample sizes: {sample_sizes}")
        
        # Also generate the overall metrics plot
        # Reshape for plotting
        overall_plot_df = pd.melt(
            overall_df, 
            id_vars=['sentence_count', 'sample_size'],
            value_vars=['accuracy', 'precision', 'recall', 'f1_score'],
            var_name='metric',
            value_name='percentage'
        )
        
        # Get unique sentence counts and their corresponding sample sizes
        unique_sentence_counts = overall_df['sentence_count'].tolist()
        sample_sizes = overall_df['sample_size'].tolist()
        
        # Plot overall metrics with transparency for better visibility of overlapping lines
        overall_plot = (
            ggplot(overall_plot_df, aes(x='sentence_count', y='percentage', color='metric', group='metric'))
            + geom_line(size=1.2, alpha=0.7)  # Add transparency with alpha
            + geom_point(size=3.5, alpha=0.8)  # Slightly larger points with transparency
            + scale_x_continuous(breaks=sentence_counts)
            + scale_y_continuous(
                limits=[0, 100],
                breaks=[0, 20, 40, 60, 80, 100],
                labels=lambda x: [f"{v}%" for v in x]
            )
            + scale_color_brewer(type='qual', palette='Set1')
            + theme_minimal()
            + theme(
                legend_position='bottom', 
                figure_size=(1.618 * 3, 3),
                panel_grid_minor=element_blank(),
                panel_grid_major_y=element_line(color='lightgray'),
                panel_grid_major_x=element_blank(),
            )
            + labs(
                title='Overall Performance Metrics by Story Length',
                x='Number of Sentences in Story',
                y='Percentage',
                color='Metric'
            )
        )
        
        # Add sample size annotations one by one
        for i, (sc, ss) in enumerate(zip(unique_sentence_counts, sample_sizes)):
            overall_plot = overall_plot + annotate('text', x=sc, y=-5, label=f"n={ss}", size=8)
        
        overall_plot_path = os.path.join(FIGURE_PATH, "overall_metrics_by_sentence_count.png")
        overall_plot.save(overall_plot_path, dpi=600)
        print(f"Saved overall metrics plot to: {overall_plot_path}")
    
    # Plot: Side-by-side comparison of all metrics for each sentence count
    # Split into multiple plots for better readability
    facet_plots = []
    
    for source_name in ['Predicted "Strange" Sentence', 'Actual Modified Sentence', 'Accuracy', 'Precision', 'Recall']:
        source_data = distribution_df[distribution_df['source'] == source_name]
        
        if source_data.empty:
            continue
            
        plot = (
            ggplot(source_data, aes(x='step_num', y='percentage', color='factor(sentence_count)', group='factor(sentence_count)'))
            + geom_line(size=1.2)
            + geom_point(size=3)
            + scale_x_continuous(breaks=[1, 3, 5, 7, 10, 15, 20], limits=[0.5, 20.5])
            + scale_y_continuous(
                limits=[0, 100],
                breaks=[0, 20, 40, 60, 80, 100],
                labels=lambda x: [f"{v}%" for v in x]
            )
            + scale_color_brewer(type='qual', palette='Accent')
            + theme_minimal()
            + theme(
                legend_position='bottom', 
                figure_size=(1.618 * 3, 3),
                panel_grid_minor=element_blank(),
                panel_grid_major_y=element_line(color='lightgray'),
                panel_grid_major_x=element_blank(),
            )
            + labs(
                title=f'{source_name} by Story Length',
                x='Sentence Number (1-20)',
                y='Percentage',
                color='Sentence Count'
            )
        )
        plot_path = os.path.join(FIGURE_PATH, f"distribution_{source_name.replace(' ', '_').replace('\"', '')}_by_sentence_count.png")
        plot.save(plot_path, dpi=600)
        print(f"Saved: {plot_path}")
        
    # Create a combined plot for precision, recall, and accuracy
    evaluation_metrics = distribution_df[distribution_df['source'].isin(['Accuracy', 'Precision', 'Recall'])]
    
    if not evaluation_metrics.empty:
        for sentence_count in sentence_counts:
            # Make sure we get all three metrics for this sentence count
            subset = evaluation_metrics[evaluation_metrics['sentence_count'] == sentence_count]
            
            # Check if we have all three metrics
            metrics_present = subset['source'].unique()
            if len(metrics_present) < 3:
                print(f"Warning: Not all metrics present for {sentence_count}-sentence stories. Found: {metrics_present}")
            
            if subset.empty:
                continue
                
            plot = (
                ggplot(subset, aes(x='step_num', y='percentage', color='source', group='source'))
                + geom_line(size=1.2)
                + geom_point(size=3)
                + scale_x_continuous(breaks=list(range(1, min(21, sentence_count + 1))), limits=[0.5, min(20.5, sentence_count + 0.5)])
                + scale_y_continuous(
                    limits=[0, 100],
                    breaks=[0, 20, 40, 60, 80, 100],
                    labels=lambda x: [f"{v}%" for v in x]
                )
                + scale_color_brewer(type='qual', palette='Set1')
                + theme_minimal()
                + theme(
                    legend_position='bottom', 
                    figure_size=(1.618 * 3, 3),
                    panel_grid_minor=element_blank(),
                    panel_grid_major_y=element_line(color='lightgray'),
                    panel_grid_major_x=element_blank(),
                )
                + labs(
                    title=f'Evaluation Metrics for {sentence_count}-Sentence Stories',
                    x='Sentence Number',
                    y='Percentage',
                    color='Metric'
                )
            )
            plot_path = os.path.join(FIGURE_PATH, f"evaluation_metrics_{sentence_count}_sentences.png")
            plot.save(plot_path, dpi=600)
            print(f"Saved: {plot_path}")

def plot_predicted_above_random(distribution_df, sentence_counts, FIGURE_PATH):
    """
    Create a plot showing the difference between predicted strange sentence distribution and random guess.
    
    Parameters:
    distribution_df (pd.DataFrame): DataFrame with distribution data including 'Predicted "Strange" Sentence' source
    sentence_counts (list): List of sentence counts to analyze
    FIGURE_PATH (str): Directory to save plot files
    """

    print("Generating predicted strange sentence above random chance plot...")
    
    # Filter only the prediction data - create an explicit copy to avoid SettingWithCopyWarning
    prediction_data = distribution_df[distribution_df['source'] == 'Predicted "Strange" Sentence'].copy()
    
    if prediction_data.empty:
        print("No prediction data available for plotting.")
        return
    
    # Calculate random guess accuracy for each sentence count
    prediction_data.loc[:, 'random_percentage'] = 100 / prediction_data['sentence_count']
    
    # Calculate difference from random
    prediction_data.loc[:, 'above_random'] = prediction_data['percentage'] - prediction_data['random_percentage']
    
    # Create plot
    plot = (
        ggplot(prediction_data, aes(x='step_num', y='above_random', color='factor(sentence_count)', group='factor(sentence_count)'))
        + geom_line(size=1.2)
        + geom_point(size=3)
        + geom_hline(yintercept=0, linetype='dashed', color='darkgray', size=0.8)
        + scale_x_continuous(breaks=[1, 3, 5, 7, 10, 15, 20], limits=[0.5, 20.5])
        + scale_color_brewer(type='qual', palette='Accent')
        + theme_minimal()
        + theme(
            legend_position='bottom', 
            figure_size=(1.618 * 3, 3),
            panel_grid_minor=element_blank(),
            panel_grid_major_y=element_line(color='lightgray'),
            panel_grid_major_x=element_blank(),
        )
        + labs(
            title='Predicted "Strange" Sentence Above Random Chance',
            x='Sentence Number (1-20)',
            y='Percentage Points Above Random Chance',
            color='Sentence Count'
        )
    )
    
    # Add annotations for random chance percentage per sentence count
    annotations = []
    for sc in sorted(prediction_data['sentence_count'].unique()):
        random_percentage = 100 / sc
        annotations.append({
            'sentence_count': sc,
            'text': f"{sc} sentences: random = {random_percentage:.1f}%"
        })
    
    # Create a list of colors for annotations
    # Use a predefined list of colors instead of trying to access the palette directly
    annotation_colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666']
    
    # Add annotation texts to the right side of the plot
    for i, row in enumerate(annotations):
        plot = plot + annotate(
            'text', 
            x=20.3, 
            y=(len(annotations) - i) * 5 - 5,  # Position from top to bottom
            label=row['text'], 
            ha='left', 
            size=8,
            color=annotation_colors[i % len(annotation_colors)]  # Use the color list safely
        )
    
    # Save plot
    plot_path = os.path.join(FIGURE_PATH, "predicted_above_random_by_position.png")
    plot.save(plot_path, dpi=600)
    print(f"Saved predicted above random plot to: {plot_path}")

if __name__ == "__main__":
    main()