import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from difflib import SequenceMatcher
from plotnine import *
from scipy import stats
import os
from typing import Tuple, List, Dict, Union, Optional

# Configuration
INPUT_DATA_PATH = "./data/step3/output_processed/"
OUTPUT_PATH = "./output/"
FIGURE_PATH = f"{OUTPUT_PATH}figure/main_result/"
TABLE_PATH = f"{OUTPUT_PATH}table/"
OUTPUT_FILE_TEMPLATE = f"{INPUT_DATA_PATH}mirror_test_results_{{}}_{{}}.csv"

# Create output directories if they don't exist
os.makedirs(FIGURE_PATH, exist_ok=True)
os.makedirs(TABLE_PATH, exist_ok=True)

# Define models
MODELS = ["ChatGPT", "Claude", "Grok", "Gemini", "Llama", "Deepseek"]

def add_confidence_interval(df: pd.DataFrame, n_samples: int = 100) -> pd.DataFrame:
    result_df = df.copy()
    result_df['ci_lower_bound'] = 0.0
    result_df['ci_upper_bound'] = 0.0

    for i, row in result_df.iterrows():
        accuracy = row['Recognition_accuracy']
        sample_size = 100
        bootstrap_samples = []
        for _ in range(n_samples):
            success_count = np.random.binomial(sample_size, accuracy)
            sample_accuracy = success_count / sample_size
            bootstrap_samples.append(sample_accuracy)

        lower, upper = np.percentile(bootstrap_samples, [2.5, 97.5])
        result_df.at[i, 'ci_lower_bound'] = lower
        result_df.at[i, 'ci_upper_bound'] = upper

    return result_df
def create_latex_table(df: pd.DataFrame, output_path: str):
    """
    Create a LaTeX table with recognition accuracy data and confidence intervals.
    This improved version ensures proper data formatting and handles edge cases.
    """
    print(f"Creating LaTeX table with {len(df)} rows of data")
    
    # Get unique model names from the dataset
    model_names = sorted(df['Model_1'].unique())
    
    # Create a dictionary to store the accuracy and CI data
    # Structure: data_dict[model1][model2] = {"acc": 0.24, "ci_lower": 0.18, "ci_upper": 0.31}
    data_dict = {}
    for model1 in model_names:
        data_dict[model1] = {}
        for model2 in model_names:
            data_dict[model1][model2] = {"acc": None, "ci_lower": None, "ci_upper": None}
    
    # Fill the dictionary with data from the DataFrame
    for _, row in df.iterrows():
        model1 = row['Model_1']
        model2 = row['Model_2']
        data_dict[model1][model2] = {
            "acc": row['Recognition_accuracy'],
            "ci_lower": row['ci_lower_bound'],
            "ci_upper": row['ci_upper_bound']
        }
    
    # Debug output to verify data is populated
    print(f"Sample data: {model_names[0]} recognizing {model_names[0]}: {data_dict[model_names[0]][model_names[0]]}")
    
    # Start building the LaTeX table
    latex_code = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\renewcommand{\\arraystretch}{1.5}",
        "\\begin{tabular}{l|" + "c" * len(model_names) + "}",
        "\\hline",
        "\\multicolumn{1}{l|}{Model 1 (story generation)} & \\multicolumn{" +
        str(len(model_names)) + "}{c}{Model 2: mark generation} \\\\",  # Note the double backslash
        "\\hline"
    ]
    
    # Add header row
    header = [" "] + model_names
    latex_code.append(" & ".join(header) + " \\\\")  # Note the double backslash
    latex_code.append("\\hline")
    
    # Add data rows
    for model1 in model_names:
        row = [model1]
        for model2 in model_names:
            data = data_dict[model1][model2]
            
            if data["acc"] is not None:
                acc = data["acc"]
                ci_lower = data["ci_lower"]
                ci_upper = data["ci_upper"]
                
                # Format data with proper precision
                acc_formatted = f"{acc*100:.1f}\\%"
                ci_formatted = f"({int(ci_lower*100)}\\%, {int(ci_upper*100)}\\%)"
                
                # Add highlighting for significant results (CI lower bound > 20%)
                highlight = "\\cellcolor{blue!15}" if ci_lower * 100 > 20 else ""
                
                # Use makecell for proper formatting
                cell = f"{highlight}\\makecell{{{acc_formatted}\\\\{ci_formatted}}}"
            else:
                cell = "-"
                
            row.append(cell)
            
        latex_code.append(" & ".join(row) + " \\\\")  # Note the double backslash
    
    # Finish the table
    latex_code.extend([
        "\\hline",
        "\\end{tabular}",
        "\\caption{Recognition Accuracy Between Models}",
        "\\label{tab:recognition-accuracy}",
        "\\end{table}"
    ])
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write("\n".join(latex_code))
    
    print(f"LaTeX table has been saved to {output_path}")
    print("Note: Include the following packages in your LaTeX preamble:")
    print("\\usepackage{makecell}\n\\usepackage{multirow}\n\\usepackage{colortbl}")

def analyze_model_accuracy():
    data_acc = []

    for model1 in MODELS:
        for model2 in MODELS:
            acc_result = f"{model1} vs {model2}"
            file_path = OUTPUT_FILE_TEMPLATE.format(model1, model2)
            try:
                df = pd.read_csv(file_path, sep="|", engine="python")
                matches_expected = sum(df['step2_random_sent_num'] == df['step3_output_int'])
                acc = matches_expected / len(df)
                is_same_model = model1 == model2
                
                # Format accuracy with percentage for output
                acc_percentage = acc * 100
                acc_result = f"{model1} vs {model2}: {acc_percentage:.2f}% accuracy"
                
                # Add special highlight for self-recognition
                if is_same_model:
                    acc_result = f"{acc_result} (self-recognition)"
                    
                print(acc_result)
                
                data_acc.append({
                    'Model_1': model1,
                    'Model_2': model2,
                    'Recognition_accuracy': round(acc, 3),
                    'Is_Same_Model': is_same_model
                })
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    df_plot = pd.DataFrame(data_acc)
    df_plot.to_csv(f"{TABLE_PATH}core_result_accuracy_table.csv", index=False)
    return df_plot

def add_significance_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Add significance level indicators to the dataframe based on statistical test against 20% baseline"""
    result_df = df.copy()
    result_df['significance'] = ''
    
    # Baseline probability of random guessing (20% or 0.2)
    baseline_prob = 0.2
    
    # Typical sample size for each test 
    sample_size = 100  # Adjust this if your actual sample size is different
    
    for i, row in result_df.iterrows():
        accuracy = row['Recognition_accuracy']
        
        # Perform binomial test against 20% baseline
        # H0: accuracy = 0.2, H1: accuracy > 0.2 (one-sided test)
        successes = int(accuracy * sample_size)
        # Use the updated scipy stats function name (binomtest instead of binom_test)
        p_value = stats.binomtest(successes, sample_size, baseline_prob, alternative='greater').pvalue
        
        # Assign significance stars based on p-value
        if p_value < 0.01:
            result_df.at[i, 'significance'] = '***'  # p < 0.01
        elif p_value < 0.05:
            result_df.at[i, 'significance'] = '**'   # p < 0.05
        elif p_value < 0.1:
            result_df.at[i, 'significance'] = '*'    # p < 0.1
    
    return result_df

def create_visualizations(df_plot: pd.DataFrame):
    # Add significance level indicators
    df_plot_with_sig = add_significance_levels(df_plot)
    
    scatter_plot = (ggplot(df_plot_with_sig, aes(x='Model_1', y='Model_2', size='Recognition_accuracy',
                                        color='Is_Same_Model'))
     + geom_point()
     + scale_color_manual(values=['#1f77b4', '#ff7f0e'])
     + theme_minimal()
     + theme(axis_text_x=element_text(angle=45, hjust=1))
     + labs(title='Mirror Test (Self-Recognition) Result',
            x='M1 (story generation and recognition)',
            y='M2 (mark generation)',
            color='Same Model'))

    scatter_plot.save(f"{FIGURE_PATH}model_recognition_scatter.png", dpi=300, width=5, height=4)

    # Create a temporary dataframe with formatted labels including significance stars
    df_heatmap = df_plot_with_sig.copy()
    df_heatmap['formatted_value'] = df_heatmap['Recognition_accuracy'].apply(lambda x: f"{x:.2f}")

    # Updated heatmap with new color scheme matching the reference image
    heatmap_plot = (ggplot(df_heatmap, aes(x='Model_2', y='Model_1', fill='Recognition_accuracy')) 
     + geom_tile(aes(width=1.0, height=1.0), color='white', size=0.5)
     + scale_fill_gradientn(
            colors=['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b'],
            values=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
            limits=[0, 1],
            name='Recognition\naccuracy')
     + geom_text(aes(label='formatted_value'), size=8, color='black')
     # Add significance stars in the top right corner of each cell
     + geom_text(
         aes(label='significance'), 
         nudge_x=0.35,            # Slight right shift
         nudge_y=0.05,           # Very close to number
         size=10,                # Bigger stars
         color='#FF4500',        # Same color
         fontweight='bold'       # Optional: bolder stars
     )
     + theme_minimal()
     + theme(axis_text_x=element_text(angle=45, hjust=1),
             panel_grid_major=element_blank(), 
             panel_grid_minor=element_blank())
     + labs(title='Mirror Test (Self-Recognition) Result',
            x='M2 (mark generation)',
            y='M1 (story generation and recognition)'))

    # For cells where Model_1 == Model_2 (self-recognition), add a dashed orange border
    same_model_data = df_heatmap[df_heatmap['Is_Same_Model'] == True]
    heatmap_plot = heatmap_plot + geom_tile(data=same_model_data, 
                            mapping=aes(x='Model_2', y='Model_1'), 
                            fill=None, color='#FF8C00', size=1.5, linetype='dashed')

    heatmap_plot.save(f"{FIGURE_PATH}model_recognition_heatmap.png", dpi=300, width=5, height=4)


def create_combined_tables_figure(df_plot_with_ci: pd.DataFrame):
    """Create a figure that shows self-recognition, non-self recognition, and mark generation statistics side by side with confidence intervals and percentage formatting"""
    # Self-recognition data
    self_rec_df = df_plot_with_ci[df_plot_with_ci['Model_1'] == df_plot_with_ci['Model_2']].copy()
    self_rec_df = self_rec_df.sort_values(by='Recognition_accuracy', ascending=False)
    self_rec_df['Type'] = 'Self-Recognition'
    
    # Non-self-recognition stats - Group by Model_1 (story generation)
    non_self_rec_df = df_plot_with_ci[df_plot_with_ci['Model_1'] != df_plot_with_ci['Model_2']].copy()
    
    # Group by Model_1 (story generation ability)
    model_stats_story = non_self_rec_df.groupby('Model_1').agg({
        'Recognition_accuracy': 'mean',
    }).reset_index()
    
    # Calculate confidence intervals for story generation ability
    for i, row in model_stats_story.iterrows():
        model = row['Model_1']
        model_data = non_self_rec_df[non_self_rec_df['Model_1'] == model]['Recognition_accuracy']
        
        # Use bootstrapping to calculate confidence intervals for the mean
        n_samples = 1000
        bootstrap_means = []
        
        for _ in range(n_samples):
            sample = np.random.choice(model_data, size=len(model_data), replace=True)
            bootstrap_means.append(np.mean(sample))
            
        lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
        model_stats_story.loc[i, 'ci_lower_bound'] = lower
        model_stats_story.loc[i, 'ci_upper_bound'] = upper
    
    model_stats_story['Type'] = 'Story Generation'
    model_stats_story = model_stats_story.sort_values(by='Recognition_accuracy', ascending=False)
    
    # Group by Model_2 (mark generation ability)
    model_stats_mark = non_self_rec_df.groupby('Model_2').agg({
        'Recognition_accuracy': 'mean',
    }).reset_index()
    
    # Rename column to maintain consistency
    model_stats_mark = model_stats_mark.rename(columns={'Model_2': 'Model_1'})
    
    # Calculate confidence intervals for mark generation ability
    for i, row in model_stats_mark.iterrows():
        model = row['Model_1']
        model_data = non_self_rec_df[non_self_rec_df['Model_2'] == model]['Recognition_accuracy']
        
        # Use bootstrapping to calculate confidence intervals for the mean
        n_samples = 1000
        bootstrap_means = []
        
        for _ in range(n_samples):
            sample = np.random.choice(model_data, size=len(model_data), replace=True)
            bootstrap_means.append(np.mean(sample))
            
        lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
        model_stats_mark.loc[i, 'ci_lower_bound'] = lower
        model_stats_mark.loc[i, 'ci_upper_bound'] = upper
    
    model_stats_mark['Type'] = 'Mark Generation'
    model_stats_mark = model_stats_mark.sort_values(by='Recognition_accuracy', ascending=False)
    
    # Convert accuracies to percentages for plotting
    self_rec_df['Recognition_accuracy_pct'] = self_rec_df['Recognition_accuracy'] * 100
    self_rec_df['ci_lower_bound_pct'] = self_rec_df['ci_lower_bound'] * 100
    self_rec_df['ci_upper_bound_pct'] = self_rec_df['ci_upper_bound'] * 100
    
    model_stats_story['Recognition_accuracy_pct'] = model_stats_story['Recognition_accuracy'] * 100
    model_stats_story['ci_lower_bound_pct'] = model_stats_story['ci_lower_bound'] * 100
    model_stats_story['ci_upper_bound_pct'] = model_stats_story['ci_upper_bound'] * 100
    
    model_stats_mark['Recognition_accuracy_pct'] = model_stats_mark['Recognition_accuracy'] * 100
    model_stats_mark['ci_lower_bound_pct'] = model_stats_mark['ci_lower_bound'] * 100
    model_stats_mark['ci_upper_bound_pct'] = model_stats_mark['ci_upper_bound'] * 100
    
    # Rename the 'Type' values before combining
    model_stats_story['Type'] = model_stats_story['Type'].replace('Story Generation', 'Mark by Model 2')
    self_rec_df['Type'] = self_rec_df['Type'].replace('Self-Recognition', 'Mark by Model 1')
    model_stats_mark['Type'] = model_stats_mark['Type'].replace('Mark Generation', 'Accuracy of other models when mark is generated by it')
    
    # Combine and plot
    combined_df = pd.concat([
        model_stats_story[['Model_1', 'Recognition_accuracy_pct', 'ci_lower_bound_pct', 'ci_upper_bound_pct', 'Type']], 
        self_rec_df[['Model_1', 'Recognition_accuracy_pct', 'ci_lower_bound_pct', 'ci_upper_bound_pct', 'Type']], 
        model_stats_mark[['Model_1', 'Recognition_accuracy_pct', 'ci_lower_bound_pct', 'ci_upper_bound_pct', 'Type']]
    ])
    
    # Create an ordered category to control bar order
    combined_df['Type'] = pd.Categorical(combined_df['Type'], 
                                       categories=['Mark by Model 2', 'Mark by Model 1', 'Accuracy of other models when mark is generated by it'], 
                                       ordered=True)
    
    # Adjust dodge width for three bars - make it wider to accommodate them
    dodge_width = 0.9
    
    # Define custom color palette to match Dark2
    custom_colors = {
        'Mark by Model 2': '#1b9e77',
        'Mark by Model 1': '#d95f02',
        'Accuracy of other models when mark is generated by it': '#7570b3'
    }
    
    # Create the plot with percentage y-axis and error bars with matching colors
    plot = (ggplot(combined_df, aes(x='Model_1', y='Recognition_accuracy_pct', fill='Type', color='Type'))
     + geom_bar(stat='identity', position=position_dodge(dodge_width), alpha=0.7)
     + geom_errorbar(
         aes(ymin='ci_lower_bound_pct', ymax='ci_upper_bound_pct'),
         position=position_dodge(dodge_width),
         width=0.25,
         size=1,
         alpha=0.6  # Make the error bars (including middle line) lighter
     )
     + geom_text(
         aes(label='Recognition_accuracy_pct'),
         format_string='{:.0f}%', 
         position=position_dodge(dodge_width),
         va='bottom',
         size=12,  # Increased from 8 to 12 to make the numbers bigger
         fontweight='bold'  # Added bold to make numbers stand out more
     )
     + scale_fill_manual(values=custom_colors)
     + scale_color_manual(values=custom_colors)
     + scale_y_continuous(labels=lambda l: [f"{v}%" for v in l])
     + theme_minimal()
     + theme(
         axis_text_x=element_text(angle=45, hjust=1),
         legend_position='bottom',  # Move legend to the bottom
         legend_text=element_text(size=11, fontweight='bold'),  # Make legend text bolder and larger
         panel_grid_minor=element_blank(),
         panel_grid_major_y=element_line(color='lightgray'),
         panel_grid_major_x=element_blank(),
        #  legend_title=element_blank(),
         legend_key_height=8,  # Make legend keys (squares) smaller  
         legend_key_width=8   # Make legend keys (squares) smaller
     )
     + labs(
         title='Model Recognition Accuracy Comparison',
         x='Model',
         y='Recognition Accuracy (%)'
     ))
    
    # Use wider dimensions to accommodate three bars per model
    plot.save(f"{FIGURE_PATH}recognition_comparison.png", dpi=300, width=12, height=5)
    print(f"Combined comparison figure saved to {FIGURE_PATH}recognition_comparison.png")

def main():
    print("\n--- Analyzing Model Accuracy ---")
    print("Calculating recognition accuracy for each model pair...")
    df_plot = analyze_model_accuracy()
    
    # Print summary of self-recognition results
    print("\nSummary of Self-Recognition Results:")
    self_rec_df = df_plot[df_plot['Is_Same_Model'] == True].sort_values(by='Recognition_accuracy', ascending=False)
    for _, row in self_rec_df.iterrows():
        model = row['Model_1']
        acc = row['Recognition_accuracy'] * 100
        print(f"{model} self-recognition: {acc:.2f}%")

    print("\n--- Creating Visualizations ---")
    create_visualizations(df_plot)

    print("\n--- Adding Confidence Intervals ---")
    df_plot_with_ci = add_confidence_interval(df_plot, n_samples=100)

    print("\n--- Creating LaTeX Tables ---")
    create_latex_table(df_plot_with_ci, f"{TABLE_PATH}recognition_accuracy_table.tex")

    # Create combined comparison figure
    print("\n--- Creating Combined Comparison Figure ---")
    create_combined_tables_figure(df_plot_with_ci)
    
    # Save processed data with all metrics
    df_plot_with_ci.to_csv(f"{TABLE_PATH}recognition_accuracy_with_ci.csv", index=False)

    print("\nAll analyses completed successfully!")
    return {
        'main_results': df_plot_with_ci
    }

if __name__ == "__main__":
    main()