#!/usr/bin/env python3
"""
plot_accuracy_by_position.py - Script to plot accuracy by sentence position for multiple evaluator models.
Using plotnine instead of matplotlib.
"""
import os
import pandas as pd
import numpy as np
from plotnine import (
    ggplot, aes, geom_line, geom_point, scale_x_continuous, 
    scale_y_continuous, scale_color_brewer, scale_fill_brewer, theme_minimal, 
    theme, element_blank, element_line, element_text, labs, 
    save_as_pdf_pages, annotate, coord_cartesian, 
    scale_color_manual, guides, guide_legend
)

# Configuration
OUTPUT_DATA_PATH_BASE = "./data/step3/"
EVALUATORS = ["claude", "gemini", "grok", "chatgpt", "llama", "deepseek"]  # Added more evaluators
EVALUATOR_LABELS = ["Claude", "Gemini", "Grok", "ChatGPT", "Llama", "Deepseek"]  # Updated labels
COLORS = ["blue", "green", "red", "purple", "orange", "teal"]  # Added more colors

# Define expected column names based on warning messages and actual column names
EXPECTED_COLUMN_PATTERNS = [
    'step3_output_int_alternative_full_sentence',  # No m3_ for full_sentence
    'step3_output_int_alternative_m3_cot',  # With m3_ for others
    'step3_output_int_alternative_m3_allow_0',
    'step3_output_int_alternative_m3_m1_unchanged',
    'step3_output_int_alternative_m3_numbered_sentences',
    'step3_output_int_alternative_m3_revealed_recognition_task'
]

# Define what's actually in the dataset
ACTUAL_COLUMN_PATTERNS = [
    'step3_output_int_alternative_full_sentence',
    'step3_output_int_alternative_cot',             # No m3_ in actual data
    'step3_output_int_alternative_allow_0',
    'step3_output_int_alternative_m1_unchanged',
    'step3_output_int_alternative_numbered_sentences',
    'step3_output_int_alternative_revealed_recognition_task'
]

# Input files - Modified to use special case for grok and handle all evaluators
INPUT_FILES = {}
for evaluator in EVALUATORS:
    if evaluator == "grok":
        INPUT_FILES[evaluator] = f"{OUTPUT_DATA_PATH_BASE}output_processed/mirror_test_results_Grok_Gemini.csv"
    else:
        INPUT_FILES[evaluator] = f"{OUTPUT_DATA_PATH_BASE}output/grok_gemini_{evaluator}/mirror_test_results_Grok_Gemini_{evaluator.capitalize()}.csv"

# Output directory for plots
FINAL_OUTPUT_DIR = "./data/step3/output/final_eval_read/"
OUTPUT_PATH = "./output/"
FIGURE_PATH = os.path.join(OUTPUT_PATH, "figure/")
M3_FOLDER = os.path.join(FIGURE_PATH, "m3/")

# Create all necessary directories
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_PATH, exist_ok=True)
os.makedirs(M3_FOLDER, exist_ok=True)

def read_evaluator_outputs():
    """
    Read the outputs from all evaluator files.
    
    Returns:
        dict: A dictionary where keys are evaluator names and values are the loaded DataFrames
    """
    outputs = {}
    
    for evaluator, file_path in INPUT_FILES.items():
        if os.path.exists(file_path):
            print(f"Reading {evaluator} output from {file_path}")
            df = pd.read_csv(file_path, sep='|')
            
            # Check for known column naming patterns
            possible_cols = [
                "step3_output_int",  # Original grok format
                f"step3_{evaluator}_output_int",  # Other evaluators format
            ]
            
            # Add all expected and actual column patterns
            possible_cols.extend(EXPECTED_COLUMN_PATTERNS)
            possible_cols.extend(ACTUAL_COLUMN_PATTERNS)
            
            # Find the first matching column
            int_col = None
            found_cols = []
            
            for col in df.columns:
                if "output_int" in col:
                    found_cols.append(col)
                    
            if found_cols:
                print(f"Found output_int columns in {evaluator} dataset: {found_cols}")
                int_col = found_cols[0]  # Use the first found column
            else:
                print(f"Warning: No output_int columns found in {evaluator} file.")
                print(f"Available columns: {df.columns.tolist()}")
                continue
            
            # Include the necessary columns
            outputs[evaluator] = df[["step2_random_sent_num", int_col]]
            outputs[evaluator]['int_col_name'] = int_col  # Store the column name we found
            print(f"Loaded {len(outputs[evaluator])} rows for {evaluator} using column '{int_col}'")
        else:
            print(f"File for {evaluator} not found: {file_path}")
    
    return outputs

def calculate_accuracy_by_position(all_outputs):
    """
    Calculate accuracy for each sentence position across all evaluators.
    
    Args:
        all_outputs: Dictionary of DataFrames with evaluator outputs
        
    Returns:
        pd.DataFrame: DataFrame with accuracy values by position for each evaluator
    """
    # Create a list to store the results for each evaluator
    results = []
    
    for evaluator, df in all_outputs.items():
        # Use the int_col_name we stored when reading the data
        int_col = df['int_col_name'].iloc[0]
        
        # Group the data by the true sentence position and calculate accuracy
        grouped = df.groupby("step2_random_sent_num")
        
        for position, group in grouped:
            # Calculate accuracy for this position
            matches = sum(group["step2_random_sent_num"] == group[int_col])
            accuracy = matches / len(group) if len(group) > 0 else 0
            
            # Append to results
            results.append({
                'evaluator': evaluator.capitalize(),  # Ensure proper capitalization
                'position': position,
                'accuracy': accuracy
            })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Print unique evaluator names to debug
    print(f"After calculation, unique evaluator names: {results_df['evaluator'].unique()}")
    
    return results_df

def plot_accuracy_by_position(accuracy_df):
    """
    Plot accuracy by sentence position for each evaluator using plotnine.
    
    Args:
        accuracy_df: DataFrame with accuracy values by position for each evaluator
    """
    # Check for NaN or missing values in evaluator column
    if accuracy_df['evaluator'].isna().any():
        print("Warning: NaN values found in evaluator column")
        print(accuracy_df[accuracy_df['evaluator'].isna()])
        # Replace NaN values with appropriate label for any problems with capitalization
        accuracy_df['evaluator'] = accuracy_df['evaluator'].fillna('Unknown')
    
    # Print unique evaluator names to debug
    print(f"Unique evaluator names in dataset: {accuracy_df['evaluator'].unique()}")
    
    # Create the original plot
    plot = (
        ggplot(accuracy_df, aes(x='position', y='accuracy', color='evaluator'))
        + geom_line(size=1.2)
        + geom_point(size=3)
        + scale_x_continuous(
            breaks=list(range(0, max(accuracy_df['position']) + 1)),
        )
        + scale_y_continuous(
            limits=[0, 1],
            breaks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=lambda x: [f"{v:.0%}" for v in x]
        )
        + scale_color_brewer(type='qual', palette='Set1')
        + theme_minimal()
        + theme(
            legend_position='bottom',
            figure_size=(1.618 * 4, 1 * 4),  # Golden ratio dimensions
            panel_grid_minor=element_blank(),
            panel_grid_major_y=element_line(color='lightgray'),
            panel_grid_major_x=element_blank(),
            legend_title=element_blank()
        )
        + labs(
            title='Accuracy by Sentence Position',
            x='Sentence Position',
            y='Accuracy',
            subtitle='Grok (Model 1) generates stories, Gemini (Model 2) marks them, all models evaluate'
        )
    )
    
    # Save the original plot
    output_path = f"{FINAL_OUTPUT_DIR}accuracy_by_position.png"
    plot.save(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    
    # Create a copy of the DataFrame for the renamed plot
    renamed_df = accuracy_df.copy()
    
    # Create a mapping dictionary for renaming
    rename_map = {
        'Claude': 'Claude',
        'Gemini': 'Gemini (Model 2)',
        'Grok': 'Grok (Model 1)',
        'Chatgpt': 'ChatGPT',
        'Llama': 'Llama',
        'Deepseek': 'Deepseek'
    }
    
    # Apply the renaming
    renamed_df['evaluator'] = renamed_df['evaluator'].map(lambda x: rename_map.get(x, x))  # Use get with default value to handle missing keys
    
    # Create the renamed plot
    renamed_plot = (
        ggplot(renamed_df, aes(x='position', y='accuracy', color='evaluator'))
        + geom_line(size=1.2)
        + geom_point(size=3)
        + scale_x_continuous(
            breaks=list(range(0, max(renamed_df['position']) + 1)),
        )
        + scale_y_continuous(
            limits=[0, 1],
            breaks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=lambda x: [f"{v:.0%}" for v in x]
        )
        + scale_color_brewer(type='qual', palette='Set1')
        + theme_minimal()
        + theme(
            legend_position='bottom',
            figure_size=(1.618 * 4, 1 * 4),  # Golden ratio dimensions
            panel_grid_minor=element_blank(),
            panel_grid_major_y=element_line(color='lightgray'),
            panel_grid_major_x=element_blank(),
            legend_title=element_blank()
        )
        + labs(
            title='Accuracy by Sentence Position',
            x='Sentence Position',
            y='Accuracy',
            subtitle='Grok (Model 1) generates stories, Gemini (Model 2) marks them, all models evaluate'
        )
    )
    
    # Save the renamed plot
    m3_output_path = os.path.join(M3_FOLDER, "m3_accuracy_by_position_renamed.png")
    renamed_plot.save(m3_output_path, dpi=300)
    print(f"Renamed plot saved to {m3_output_path}")
    
    # Create a subset plot with only the top evaluators
    # Determine which evaluators to include in a top performers plot
    top_evaluators = get_top_evaluators(accuracy_df, 3)  # Get top 3 evaluators
    top_df = accuracy_df[accuracy_df['evaluator'].isin(top_evaluators)]
    
    top_plot = (
        ggplot(top_df, aes(x='position', y='accuracy', color='evaluator'))
        + geom_line(size=1.2)
        + geom_point(size=3)
        + scale_x_continuous(
            breaks=list(range(0, max(top_df['position']) + 1)),
        )
        + scale_y_continuous(
            limits=[0, 1],
            breaks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=lambda x: [f"{v:.0%}" for v in x]
        )
        + scale_color_brewer(type='qual', palette='Set1')
        + theme_minimal()
        + theme(
            legend_position='bottom',
            figure_size=(1.618 * 4, 1 * 4),  # Golden ratio dimensions
            panel_grid_minor=element_blank(),
            panel_grid_major_y=element_line(color='lightgray'),
            panel_grid_major_x=element_blank(),
            legend_title=element_blank()
        )
        + labs(
            title='Top Performers: Accuracy by Sentence Position',
            x='Sentence Position',
            y='Accuracy',
            subtitle='Showing the best performing models in the evaluation task'
        )
    )
    
    # Save the top performers plot
    top_output_path = os.path.join(FIGURE_PATH, "top_evaluators_accuracy.png")
    top_plot.save(top_output_path, dpi=300)
    print(f"Top evaluators plot saved to {top_output_path}")
    
    return plot, renamed_plot, top_plot

def get_top_evaluators(accuracy_df, n=3):
    """
    Get the top n evaluators based on overall accuracy
    
    Args:
        accuracy_df: DataFrame with accuracy values
        n: Number of top evaluators to return
        
    Returns:
        list: List of top evaluator names
    """
    # Calculate average accuracy for each evaluator
    evaluator_accuracy = accuracy_df.groupby('evaluator')['accuracy'].mean().reset_index()
    
    # Sort by accuracy in descending order and get top n
    top_evaluators = evaluator_accuracy.sort_values('accuracy', ascending=False).head(n)['evaluator'].tolist()
    
    return top_evaluators

def calculate_overall_accuracy(all_outputs):
    """
    Calculate overall accuracy for each evaluator.
    
    Args:
        all_outputs: Dictionary of DataFrames with evaluator outputs
        
    Returns:
        dict: A dictionary with overall accuracy for each evaluator
    """
    overall_accuracy = {}
    
    for evaluator, df in all_outputs.items():
        # Use the int_col_name we stored when reading the data
        int_col = df['int_col_name'].iloc[0]
        
        # Calculate overall accuracy
        matches = sum(df["step2_random_sent_num"] == df[int_col])
        accuracy = matches / len(df) if len(df) > 0 else 0
        overall_accuracy[evaluator] = accuracy
    
    return overall_accuracy

def save_accuracy_comparison_table(overall_accuracy):
    """
    Save a CSV table with accuracy comparison for all evaluators.
    
    Args:
        overall_accuracy: Dictionary with overall accuracy for each evaluator
    """
    # Create a DataFrame from the dictionary
    comparison_df = pd.DataFrame({
        'Evaluator': [k.capitalize() for k in overall_accuracy.keys()],
        'Overall Accuracy': [overall_accuracy[k] for k in overall_accuracy.keys()]
    })
    
    # Sort by accuracy in descending order
    comparison_df = comparison_df.sort_values('Overall Accuracy', ascending=False)
    
    # Format accuracy as percentage
    comparison_df['Overall Accuracy'] = comparison_df['Overall Accuracy'].apply(lambda x: f"{x:.2%}")
    
    # Save to CSV
    output_path = f"{FINAL_OUTPUT_DIR}evaluator_accuracy_comparison.csv"
    comparison_df.to_csv(output_path, index=False)
    print(f"Accuracy comparison table saved to {output_path}")
    
    return comparison_df

def main():
    # Load evaluator outputs
    all_outputs = read_evaluator_outputs()
    
    if not all_outputs:
        print("No evaluator outputs found. Please check input file paths.")
        return
    
    # Calculate accuracy by position and convert to DataFrame
    accuracy_df = calculate_accuracy_by_position(all_outputs)
    
    # Calculate overall accuracy
    overall_accuracy = calculate_overall_accuracy(all_outputs)
    print("\nOverall Accuracy by Evaluator:")
    for evaluator, accuracy in overall_accuracy.items():
        print(f"{evaluator.capitalize()}: {accuracy:.2%}")
    
    # Save accuracy comparison table
    comparison_table = save_accuracy_comparison_table(overall_accuracy)
    print("\nEvaluator Accuracy Comparison:")
    print(comparison_table)
    
    # Plot accuracy by position (returns both original and renamed plots)
    original_plot, renamed_plot, top_plot = plot_accuracy_by_position(accuracy_df)
    
    # You can display the plots in a Jupyter notebook with:
    # print(original_plot)
    # print(renamed_plot)
    # print(top_plot)

if __name__ == "__main__":
    main()