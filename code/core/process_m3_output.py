import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.util import *
import pandas as pd
import glob
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from difflib import SequenceMatcher
from nltk.tokenize import sent_tokenize
import argparse

# Constants 
OUTPUT_DATA_PATH = "./data/step3/output/"
OUTPUT_FILE_TEMPLATE = f"./data/step3/output_processed/mirror_test_results_{{}}_{{}}.csv"

# Models list
models = ["ChatGPT", "Claude", "Grok", "Gemini", "Llama", "Deepseek"]

def read_all_csv_results() -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    Read all CSV output files from the mirror test analysis.
    
    Returns:
        Dictionary mapping model pairs to their respective dataframes
    """
    results = {}
    
    # Check if the output directory exists
    if not os.path.exists(OUTPUT_DATA_PATH):
        print(f"Output directory {OUTPUT_DATA_PATH} does not exist.")
        return results
    
    # Find all CSV files in the output directory
    csv_files = glob.glob(f"{OUTPUT_DATA_PATH}*.csv")
    print(f"Found {len(csv_files)} CSV files in {OUTPUT_DATA_PATH}")
    
    if not csv_files:
        print("No CSV files found with the expected naming pattern.")
        return results
    
    # Process each file
    for file_path in csv_files:
        # Extract model names from filename
        filename = os.path.basename(file_path)
        # Fix: Use string method on string variable
        if not isinstance(filename, str) or not filename.startswith("mirror_test_results_"):
            continue
            
        # Parse filename to extract model pair
        try:
            # Remove "mirror_test_results_" prefix and ".csv" suffix
            model_part = filename[len("mirror_test_results_"):-4]
            # Split by underscore to get the model names
            model1, model2 = model_part.split("_")
            
            # Read the CSV file
            df = pd.read_csv(file_path, sep='|')
            
            results[(model1, model2)] = df
            print(f"Loaded {len(df)} records from {model1} vs {model2}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    return results

def process_dataframes(results: Dict[Tuple[str, str], pd.DataFrame]) -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    Process each DataFrame to extract numeric predictions and calculate accuracy.
    
    Args:
        results: Dictionary mapping model pairs to their dataframes
        
    Returns:
        Dictionary of processed dataframes
    """
    processed_results = {}
    
    for (model1, model2), df in results.items():
        print(f"\nProcessing {model1} vs {model2}...")
        try:
            # Create a copy to avoid modifying the original
            processed_df = df.copy()
            
            output_column = 'step3_output_message_only'
            
            # Check if the output column exists
            if output_column in processed_df.columns:
                # Convert output message to string explicitly
                processed_df[output_column] = processed_df[output_column].fillna('').astype(str)
                
                # Extract numeric value from the output message
                processed_df['step3_output_int'] = pd.to_numeric(
                    processed_df[output_column].str.extract(r'(\d+)').iloc[:, 0],
                    errors='coerce'
                ).fillna(0).astype(int)
            else:
                print(f"Warning: Output column '{output_column}' not found in {model1} vs {model2}")
            
            # Make sure step2_random_sent_num is numeric if it exists
            if 'step2_random_sent_num' in processed_df.columns:
                processed_df['step2_random_sent_num'] = pd.to_numeric(
                    processed_df['step2_random_sent_num'], 
                    errors='coerce'
                ).fillna(0).astype(int)
            
            # Calculate if the prediction is correct
            if 'step3_output_int' in processed_df.columns and 'step2_random_sent_num' in processed_df.columns:
                processed_df['is_correct'] = (processed_df['step3_output_int'] == processed_df['step2_random_sent_num'])
                
                # Calculate accuracy
                accuracy = processed_df['is_correct'].mean()
                print(f"Accuracy for {model1} vs {model2}: {accuracy:.2%}")
            
            # Save to processed results
            processed_results[(model1, model2)] = processed_df
            
            output_file = OUTPUT_FILE_TEMPLATE.format(model1, model2)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            processed_df.to_csv(output_file, sep='|', index=False)
            print(f"Processed results saved to {output_file}")
            
        except Exception as e:
            print(f"Error processing {model1} vs {model2}: {e}")
    
    return processed_results

def generate_summary_stats(results: Dict[Tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """
    Generate summary statistics for all model pairs.
    
    Args:
        results: Dictionary mapping model pairs to their dataframes
        
    Returns:
        DataFrame with summary statistics
    """
    summary_data = []
    
    for (model1, model2), df in results.items():
        # Extract basic stats
        record_count = len(df)
        
        try:
            # Calculate accuracy if 'is_correct' column exists
            if 'is_correct' in df.columns:
                accuracy = df['is_correct'].mean()
            else:
                accuracy = None
            
            output_column = 'step3_output_message_only'
                
            # Calculate average response length - ensure we're working with strings
            if output_column in df.columns:
                # Fill NAs and convert to string
                strings_only = df[output_column].fillna('').astype(str)
                avg_response_length = strings_only.str.len().mean()
            else:
                avg_response_length = None
                
            summary_data.append({
                'Model1': model1,
                'Model2': model2,
                'Records': record_count,
                'Accuracy': accuracy,
                'Avg Response Length': avg_response_length,
            })
        except Exception as e:
            print(f"Error calculating stats for {model1} vs {model2}: {e}")
    
    return pd.DataFrame(summary_data)

def visualize_model_comparisons(summary_df: pd.DataFrame):
    """
    Create visualizations for model comparisons.
    
    Args:
        summary_df: DataFrame with summary statistics
    """
    # Ensure output directory exists
    os.makedirs("output/temp", exist_ok=True)
    
    # Set up plotting
    plt.figure(figsize=(15, 10))
    sns.set(style="whitegrid")
    
    # Bar chart of record counts by model pair
    plt.subplot(2, 1, 1)
    chart = sns.barplot(x='Model1', y='Records', hue='Model2', data=summary_df)
    plt.title('Number of Records by Model Pair')
    plt.xticks(rotation=45)
    
    # Bar chart of accuracy by model pair (only if accuracy column exists and has values)
    if 'Accuracy' in summary_df.columns and not summary_df['Accuracy'].isna().all():
        plt.subplot(2, 1, 2)
        chart = sns.barplot(x='Model1', y='Accuracy', hue='Model2', data=summary_df)
        plt.title('Accuracy by Model Pair')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = os.path.join("output/temp", 'model_comparison_summary.png')
    plt.savefig(plot_filename)
    print(f"Visualization saved to {plot_filename}")
    
    # Create a heatmap of model accuracy (only if accuracy column exists and has values)
    if 'Accuracy' in summary_df.columns and not summary_df['Accuracy'].isna().all():
        try:
            plt.figure(figsize=(10, 8))
            
            # Prepare data for heatmap
            heatmap_data = summary_df.pivot(index='Model1', columns='Model2', values='Accuracy')
            
            # Plot heatmap
            sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2%")
            plt.title('Mirror Test Accuracy Heatmap')
            plt.tight_layout()
            
            # Save heatmap
            heatmap_filename = os.path.join("output/temp", 'accuracy_heatmap.png')
            plt.savefig(heatmap_filename)
            print(f"Accuracy heatmap saved to {heatmap_filename}")
        except Exception as e:
            print(f"Error creating accuracy heatmap: {e}")

def analyze_model_performance(results: Dict[Tuple[str, str], pd.DataFrame]):
    """
    Perform advanced analysis on model performance.
    
    Args:
        results: Dictionary mapping model pairs to their processed dataframes
    """
    # Ensure output directory exists
    os.makedirs("output/temp", exist_ok=True)
    
    # Create aggregated stats by model
    model_stats = {}
    
    # For each model, collect all instances where it was tested
    for (model1, model2), df in results.items():
        if 'is_correct' not in df.columns:
            continue
            
        # Update stats for model1
        if model1 not in model_stats:
            model_stats[model1] = {'total_correct': 0, 'total_tests': 0}
        
        model_stats[model1]['total_correct'] += df['is_correct'].sum()
        model_stats[model1]['total_tests'] += len(df)
        
        # Update stats for model2
        if model2 not in model_stats:
            model_stats[model2] = {'total_correct': 0, 'total_tests': 0}
        
        # For model2, we invert the correctness (assuming mirror test)
        model_stats[model2]['total_correct'] += len(df) - df['is_correct'].sum()
        model_stats[model2]['total_tests'] += len(df)
    
    # Calculate overall accuracy for each model
    for model, stats in model_stats.items():
        if stats['total_tests'] > 0:
            stats['overall_accuracy'] = stats['total_correct'] / stats['total_tests']
        else:
            stats['overall_accuracy'] = None
    
    # Convert to DataFrame for easier visualization
    model_performance_df = pd.DataFrame([
        {
            'Model': model,
            'Total Tests': stats['total_tests'],
            'Total Correct': stats['total_correct'],
            'Overall Accuracy': stats['overall_accuracy']
        }
        for model, stats in model_stats.items()
    ])
    
    # Sort by accuracy
    model_performance_df = model_performance_df.sort_values('Overall Accuracy', ascending=False)
    
    # Display and save results
    print("\nOverall Model Performance:")
    print(model_performance_df)
    
    # Save performance summary
    performance_csv = os.path.join("output/temp", 'model_overall_performance.csv')
    model_performance_df.to_csv(performance_csv, index=False)
    print(f"Model performance summary saved to {performance_csv}")
    
    # Visualize overall model performance
    plt.figure(figsize=(10, 6))
    
    # Plot accuracy by model
    chart = sns.barplot(x='Model', y='Overall Accuracy', data=model_performance_df)
    plt.title('Overall Model Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.0)  # Set y-axis from 0 to 1
    
    # Add value labels on bars
    for p in chart.patches:
        chart.annotate(f"{p.get_height():.2%}", 
                      (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha = 'center', va = 'bottom',
                      xytext = (0, 5), textcoords = 'offset points')
    
    plt.tight_layout()
    
    # Save chart
    chart_filename = os.path.join("output/temp", 'overall_model_accuracy.png')
    plt.savefig(chart_filename)
    print(f"Overall model accuracy chart saved to {chart_filename}")

def main():
    # Ensure output directory exists
    os.makedirs("output/temp", exist_ok=True)
    
    # Read all CSV results
    results = read_all_csv_results()
    
    if not results:
        print("No results to analyze.")
        return
    
    print(f"Successfully loaded data for {len(results)} model pairs.")
    
    # Process dataframes to extract numeric predictions and calculate accuracy
    processed_results = process_dataframes(results)
    
    # Generate summary statistics
    summary_df = generate_summary_stats(processed_results)
    print("\nSummary Statistics:")
    print(summary_df)
    
    # Save summary to CSV
    summary_csv = os.path.join("output/temp", 'mirror_test_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary saved to {summary_csv}")
    
    # Create visualizations
    try:
        visualize_model_comparisons(summary_df)
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    # Perform advanced analysis
    try:
        analyze_model_performance(processed_results)
    except Exception as e:
        print(f"Error in advanced analysis: {e}")
    
    # Find model pairs with the most records
    if not summary_df.empty:
        max_records = summary_df['Records'].max()
        most_records = summary_df[summary_df['Records'] == max_records]
        print(f"\nModel pairs with the most records ({max_records}):")
        print(most_records[['Model1', 'Model2']])
    
    # Find model pairs with the highest accuracy
    if 'Accuracy' in summary_df.columns and not summary_df['Accuracy'].isna().all():
        max_accuracy = summary_df['Accuracy'].max()
        highest_accuracy = summary_df[summary_df['Accuracy'] == max_accuracy]
        print(f"\nModel pairs with the highest accuracy ({max_accuracy:.2%}):")
        print(highest_accuracy[['Model1', 'Model2', 'Accuracy']])

if __name__ == "__main__":
    main()