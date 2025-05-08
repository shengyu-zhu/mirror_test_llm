#!/usr/bin/env python3
"""
Script to read the merged data from OUTPUT_DIR/merged_all_variants_by_key.csv and
merge it with step3/output_processed/mirror_test_results_Grok_Gemini.csv using
the key variable step2_output_nth_sentence_message_only.

Updated to work with the simplified merge_data_m3_prompt_variants.py script.
"""
import os
import pandas as pd
import glob
from tqdm import tqdm
import numpy as np
import sys
import json

# Define constants
BASE_DIR = './data/step3/'
OUTPUT_DIR = './data/step3/merged/'
MERGED_FILE = f"{OUTPUT_DIR}merged_all_variants_by_key.csv"
OUTPUT_FILE = f"{BASE_DIR}output_processed/mirror_test_results_Grok_Gemini.csv"
FINAL_OUTPUT_FILE = f"{OUTPUT_DIR}final_merged_results.csv"

# Define paths for figures
OUTPUT_PATH = "./output/figure"
# Additional paths for variant-specific figures
VARIANTS_PATH = os.path.join(OUTPUT_PATH, "m3_prompt_alternatives/")
HISTOGRAM_PATH = os.path.join(VARIANTS_PATH, "histogram/")

# Create output directories
os.makedirs(VARIANTS_PATH, exist_ok=True)
os.makedirs(HISTOGRAM_PATH, exist_ok=True)

# Check if required plotting libraries are available
PLOTTING_AVAILABLE = False
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from plotnine import *
    PLOTTING_AVAILABLE = True
except ImportError:
    print("Warning: Plotting libraries not available. Install with: pip install matplotlib seaborn plotnine")

def check_variant_summaries():
    """
    Check if variant summaries exist in the expected directories.
    This is useful to determine if we're working with real data or placeholders.
    
    Returns:
        dict: Dictionary with variant names as keys and True/False for summary existence
    """
    result = {}
    
    for variant in [
        "alternative_m3_full_sentence",
        "alternative_m3_cot",
        "alternative_m3_allow_0",
        "alternative_m3_m1_unchanged", 
        "alternative_m3_numbered_sentences",
        "alternative_m3_revealed_recognition_task"
    ]:
        # Check if summaries directory exists
        variant_path = f"{BASE_DIR}output_{variant.replace('alternative_m3_', '')}/"
        summary_dir = f"{variant_path}summaries/"
        
        # Check if any summary files exist
        if os.path.exists(summary_dir):
            summary_files = glob.glob(f"{summary_dir}summary_*.json")
            result[variant] = len(summary_files) > 0
        else:
            result[variant] = False
    
    return result

def load_variant_summaries():
    """
    Load any available variant summaries to use in reporting.
    
    Returns:
        dict: Dictionary with variant names as keys and summary data as values
    """
    summaries = {}
    
    for variant in [
        "alternative_m3_full_sentence",
        "alternative_m3_cot",
        "alternative_m3_allow_0", 
        "alternative_m3_m1_unchanged",
        "alternative_m3_numbered_sentences",
        "alternative_m3_revealed_recognition_task"
    ]:
        # Check if summaries directory exists
        variant_path = f"{BASE_DIR}output_{variant.replace('alternative_m3_', '')}/"
        summary_dir = f"{variant_path}summaries/"
        
        # Check for Grok-Gemini summary specifically
        summary_file = f"{summary_dir}summary_Grok_Gemini_{variant}.json"
        
        if os.path.exists(summary_file):
            try:
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)
                summaries[variant] = summary_data
                print(f"Loaded summary for {variant}: accuracy = {summary_data.get('accuracy', 0):.2%}")
            except Exception as e:
                print(f"Error loading summary for {variant}: {e}")
    
    return summaries

def create_accuracy_visualizations(accuracy_by_sentence):
    """
    Create visualizations for accuracy by sentence position.
    
    Args:
        accuracy_by_sentence: List of dictionaries with accuracy data
    """
    if not accuracy_by_sentence:
        print("No accuracy data available for visualizations")
        return
    
    try:
        # Convert to DataFrame for plotting
        acc_by_sent_df = pd.DataFrame(accuracy_by_sentence)
        
        # Define mapping from internal variant names to clearer labels
        variant_name_mapping = {
            'standard': 'Original design',
            'alternative_full_sentence': 'Output full sentence',
            'alternative_cot': 'Chain of thought',
            'alternative_allow_0': 'Allow answer 0',
            'alternative_m1_unchanged': 'Original step1 story',
            'alternative_numbered_sentences': 'Numbered sentences',
            'alternative_revealed_recognition_task': 'Reveal task objective'
        }
        
        # Map variant column to the clearer label
        acc_by_sent_df['variant_label'] = acc_by_sent_df['variant'].map(
            lambda x: variant_name_mapping.get(x, x)
        )
        
        # ===== PLOTNINE VERSION =====
        # Create the plot using plotnine
        plot_accuracy = (
            ggplot(acc_by_sent_df, aes(x='sentence_number', y='accuracy', color='variant_label', group='variant_label')) +
            geom_line(size=1.2) +
            geom_point(size=3) +
            scale_x_continuous(
                breaks=[1, 2, 3, 4, 5],
                limits=[0.8, 5.2]
            ) +
            scale_y_continuous(
                limits=[0, 100],
                breaks=list(range(0, 101, 20)),
                labels=lambda x: [f"{v}%" for v in x]
            ) +
            scale_color_brewer(type='qual', palette='Dark2') +
            theme_minimal() +
            theme(
                legend_position='bottom',
                figure_size=(1.618 * 4, 1 * 4),
                panel_grid_minor=element_blank(),
                panel_grid_major_y=element_line(color='lightgray'),
                panel_grid_major_x=element_blank(),
                legend_title=element_blank()
            ) +
            labs(
                title='Sentence Recognition Accuracy by Position',
                x='Sentence Number',
                y='Accuracy Percentage'
            )
        )
        
        # Save the plot
        accuracy_plot_path = os.path.join(VARIANTS_PATH, "sentence_accuracy_by_position.png")
        plot_accuracy.save(accuracy_plot_path, dpi=300)
        print(f"Saved accuracy by position plot to: {accuracy_plot_path}")
        
        # ===== MATPLOTLIB VERSION =====
        # Using matplotlib for alternative visualization with percentage labels
        plt.figure(figsize=(12, 7))
        
        # Plot a line for each variant
        colors = plt.cm.Dark2(np.linspace(0, 1, len(acc_by_sent_df['variant_label'].unique())))
        
        for i, variant in enumerate(acc_by_sent_df['variant_label'].unique()):
            # Get data for this variant
            variant_data = acc_by_sent_df[acc_by_sent_df['variant_label'] == variant]
            
            # Sort by sentence number
            variant_data = variant_data.sort_values('sentence_number')
            
            # Plot the line
            plt.plot(
                variant_data['sentence_number'], 
                variant_data['accuracy'],
                marker='o',
                linewidth=2,
                label=variant,
                color=colors[i]
            )
            
            # Add accuracy labels
            for _, row in variant_data.iterrows():
                plt.text(
                    row['sentence_number'], 
                    row['accuracy'] + 2,
                    f"{row['accuracy']:.1f}%",
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    color=colors[i]
                )
        
        # Customize the plot
        plt.xlabel('Sentence Number')
        plt.ylabel('Accuracy (%)')
        plt.title('Sentence Recognition Accuracy by Position and Variant')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(1, 6))
        plt.ylim(0, 100)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save the plot
        mpl_accuracy_plot_path = os.path.join(VARIANTS_PATH, 'sentence_accuracy_percentage.png')
        plt.savefig(mpl_accuracy_plot_path, dpi=300)
        print(f"Saved matplotlib version to: {mpl_accuracy_plot_path}")
        
        # Create a table of accuracies by sentence position
        print("\nAccuracy by Sentence Position:")
        
        # Pivot the dataframe to create a table with variants as rows and sentence numbers as columns
        pivot_df = acc_by_sent_df.pivot(index='variant_label', columns='sentence_number', values='accuracy')
        
        # Add a mean column
        pivot_df['Mean'] = pivot_df.mean(axis=1)
        
        # Format the table with percentages
        formatted_pivot = pivot_df.applymap(lambda x: f"{x:.1f}%")
        
        # Print the table
        print(formatted_pivot)
        
        # Save the table as a CSV
        table_path = os.path.join(VARIANTS_PATH, 'sentence_position_accuracy_table.csv')
        pivot_df.to_csv(table_path)
        print(f"Saved accuracy table to: {table_path}")
        
    except Exception as e:
        print(f"Error creating accuracy visualizations: {e}")
        import traceback
        traceback.print_exc()

def create_distribution_visualizations(column_counts, accuracy_results=None):
    """
    Create visualizations for the distribution of predicted sentence numbers.
    
    Args:
        column_counts: Dictionary with column counts
        accuracy_results: Optional list of dictionaries with accuracy results
    """
    if not column_counts:
        print("No column counts available for distribution visualizations")
        return
    
    try:
        # Calculate percentages for all variants
        percentage_data = []
        
        for col in column_counts.keys():
            # Get total count
            total = sum(column_counts[col].values)
            
            if total > 0:
                # Calculate percentage for each value
                for sent_num, count in column_counts[col].items():
                    # Ensure sent_num is an integer
                    sent_num_int = int(sent_num)
                    
                    # Skip if not in range 1-5 (or 0-5 for allow_0)
                    if sent_num_int in range(6):  # Includes 0-5
                        percentage = (count / total) * 100
                        
                        # Add to data for plotting
                        variant_name = col.replace('step3_output_int_', '').replace('step3_output_int', 'standard')
                        percentage_data.append({
                            'variant': variant_name,
                            'step_num': sent_num_int,
                            'percentage': percentage
                        })
        
        # Convert to DataFrame for plotting
        if percentage_data:
            pct_df = pd.DataFrame(percentage_data)
            
            # Ensure step_num is an integer
            pct_df['step_num'] = pct_df['step_num'].astype(int)
            
            # Filter out empty rows and make sure to include 0 values
            pct_df = pct_df[pct_df['step_num'].notna()]
            
            # If we have data, create the plot
            if not pct_df.empty:
                # Calculate max value for y-axis limit (with no extra headroom)
                max_percentage = pct_df['percentage'].max()
                y_limit = max_percentage * 1.0
                
                filtered_df = pct_df[pct_df['percentage'] >= 0.2]

                # Define mapping from internal variant names to clearer labels
                variant_name_mapping = {
                    'standard': 'Original design',
                    'alternative_full_sentence': 'Output full sentence',
                    'alternative_cot': 'Chain of thought',
                    'alternative_allow_0': 'Allow answer 0',
                    'alternative_m1_unchanged': 'Original step1 story',
                    'alternative_numbered_sentences': 'Numbered sentences',
                    'alternative_revealed_recognition_task': 'Reveal task objective'
                }
                
                # Map variant column to the clearer label
                filtered_df['variant'] = filtered_df['variant'].map(
                    lambda x: variant_name_mapping.get(x, x)
                )

                # Create the plot using plotnine
                plot1 = (
                    ggplot(filtered_df, aes(x='step_num', y='percentage', color='variant', group='variant')) +
                    geom_line(size=1.2) +
                    geom_point(size=3) +
                    scale_x_continuous(
                        breaks=[0, 1, 2, 3, 4, 5],
                        limits=[-0.2, 5.2]
                    ) +
                    scale_y_continuous(
                        limits=[0, y_limit],
                        breaks=list(range(0, int(y_limit)+1, 10)),
                        labels=lambda x: [f"{v}%" for v in x]
                    ) +
                    scale_color_brewer(type='qual', palette='Dark2') +
                    theme_minimal() +
                    theme(
                        legend_position='bottom',
                        figure_size= (1.618 * 3.8, 1* 3.8),
                        panel_grid_minor=element_blank(),
                        panel_grid_major_y=element_line(color='lightgray'),
                        panel_grid_major_x=element_blank(),
                        legend_title=element_blank()
                    ) +
                    labs(
                        title='Sentence Number Distribution by Variant',
                        x='Predicted Sentence Number',
                        y='Percentage'
                    )
                )
                
                # Save the plot to appropriate locations
                # Save to the variants folder
                variants_plot_path = os.path.join(VARIANTS_PATH, "sentence_distribution.png")
                plot1.save(variants_plot_path, dpi=300)
                print(f"Saved: {variants_plot_path}")
                
                # Now create individual plots for each variant using matplotlib
                variants = pct_df['variant'].unique()
                
                # ===== MATPLOTLIB VERSION (sentence_distribution_percentage.png) =====
                plt.figure(figsize=(10, 6))
                
                # Plot a line for each variant
                colors = plt.cm.tab10(np.linspace(0, 1, len(variants)))
                
                for i, variant in enumerate(variants):
                    # Get data for this variant
                    variant_data = pct_df[pct_df['variant'] == variant]
                    
                    # Sort by sentence number
                    variant_data = variant_data.sort_values('step_num')
                    
                    # Plot the line
                    plt.plot(
                        variant_data['step_num'], 
                        variant_data['percentage'],
                        marker='o',
                        linewidth=2,
                        label=variant_name_mapping.get(variant, variant),
                        color=colors[i]
                    )
                    
                    # Add percentage labels
                    for _, row in variant_data.iterrows():
                        plt.text(
                            row['step_num'], 
                            row['percentage'] + 1,
                            f"{row['percentage']:.1f}%",
                            ha='center',
                            va='bottom',
                            fontsize=8,
                            color=colors[i]
                        )
                
                # Customize the plot
                plt.xlabel('Predicted Sentence Number')
                plt.ylabel('Percentage (%)')
                plt.title('Distribution of Predicted Sentence Numbers by Variant')
                plt.grid(True, alpha=0.3)
                plt.xticks(range(6))  # Include 0-5
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                
                # Save the plot only to variants folder
                pct_plot_path = os.path.join(VARIANTS_PATH, 'sentence_distribution_percentage.png')
                plt.savefig(pct_plot_path, dpi=300)
                print(f"Saved percentage distribution plot to: {pct_plot_path}")
                
                # Create individual histogram plots for each variant
                for variant in variants:
                    variant_name = variant_name_mapping.get(variant, variant)
                    plt.figure(figsize=(8, 5))
                    
                    # Get data for this variant
                    variant_data = pct_df[pct_df['variant'] == variant]
                    
                    # Sort by sentence number
                    variant_data = variant_data.sort_values('step_num')
                    
                    # Plot the bars
                    bars = plt.bar(
                        variant_data['step_num'],
                        variant_data['percentage'],
                        color='skyblue'
                    )
                    
                    # Add percentage labels
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(
                            bar.get_x() + bar.get_width()/2.,
                            height + 0.5,
                            f"{height:.1f}%",
                            ha='center',
                            va='bottom'
                        )
                    
                    # Customize the plot
                    plt.xlabel('Predicted Sentence Number')
                    plt.ylabel('Percentage (%)')
                    plt.title(f'Distribution of Predicted Sentence Numbers - {variant_name}')
                    plt.grid(axis='y', alpha=0.3)
                    
                    # Set x-ticks based on the variant
                    if variant == 'alternative_allow_0':
                        plt.xticks(range(6))  # Include 0-5
                    else:
                        plt.xticks(range(1, 6))  # Only 1-5
                        
                    plt.ylim(0, max(variant_data['percentage']) * 1.15)
                    plt.tight_layout()
                    
                    # Save the plot
                    variant_path = os.path.join(HISTOGRAM_PATH, f'sentence_distribution_{variant}.png')
                    plt.savefig(variant_path, dpi=300)
                    print(f"Saved individual plot for {variant}")
                
                # Create accuracy bar plot based on results
                # Calculate accuracy for each variant
                if accuracy_results:
                    accuracy_data = []
                    for col, counts in column_counts.items():
                        variant_name = col.replace('step3_output_int_', '').replace('step3_output_int', 'standard')
                        
                        # Find this variant in the accuracy results
                        for result in accuracy_results:
                            if result['Column'] == col:
                                accuracy_data.append({
                                    'variant': variant_name,
                                    'accuracy': result['Accuracy (%)']
                                })
                                break
                    
                    # Create DataFrame for accuracy plot
                    if accuracy_data:
                        acc_df = pd.DataFrame(accuracy_data)
                        
                        # Sort by accuracy (descending)
                        acc_df = acc_df.sort_values('accuracy', ascending=False)
                        
                        # Map to friendly names
                        acc_df['variant_label'] = acc_df['variant'].map(
                            lambda x: variant_name_mapping.get(x, x)
                        )
                        
                        # Create bar chart
                        plot2 = (
                            ggplot(acc_df, aes(x='variant_label', y='accuracy', fill='variant_label'))
                            + geom_bar(stat='identity', position='dodge')
                            + geom_text(aes(label='accuracy'), 
                                       format_string='{:.1f}%',
                                       position=position_dodge(width=0.9), 
                                       va='bottom')
                            + scale_y_continuous(
                                limits=[0, max(acc_df['accuracy']) * 1.15],
                                breaks=range(0, 101, 10),
                                labels=lambda x: [f"{v}%" for v in x]
                            )
                            + theme_minimal()
                            + theme(
                                axis_text_x=element_text(angle=45, hjust=1),
                                figure_size=(10, 6),
                                panel_grid_minor=element_blank(),
                                panel_grid_major_x=element_blank(),
                                legend_position='none'
                            )
                            + labs(
                                title='Accuracy by Variant',
                                x='Variant',
                                y='Accuracy Percentage'
                            )
                        )
                        
                        # Save the accuracy plot to variants folder
                        plot2_path = os.path.join(VARIANTS_PATH, "variant_accuracy_comparison.png")
                        plot2.save(plot2_path, dpi=300)
                        print(f"Saved: {plot2_path}")
                        
                        # Also create a matplotlib version
                        plt.figure(figsize=(10, 6))
                        
                        # Plot the bars
                        bars = plt.bar(
                            acc_df['variant_label'],
                            acc_df['accuracy'],
                            color='skyblue'
                        )
                        
                        # Add percentage labels
                        for bar in bars:
                            height = bar.get_height()
                            plt.text(
                                bar.get_x() + bar.get_width()/2.,
                                height + 1,
                                f"{height:.1f}%",
                                ha='center',
                                va='bottom'
                            )
                        
                        # Customize the plot
                        plt.xlabel('Variant')
                        plt.ylabel('Accuracy (%)')
                        plt.title('Sentence Recognition Accuracy by Variant')
                        plt.grid(axis='y', alpha=0.3)
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        
                        # Save the plot
                        mpl_acc_path = os.path.join(VARIANTS_PATH, 'variant_accuracy_mpl.png')
                        plt.savefig(mpl_acc_path, dpi=300)
                        print(f"Saved matplotlib accuracy plot to: {mpl_acc_path}")
                
                # Create histograms for each variant
                try:
                    # Create a histogram for each valid column
                    valid_columns = list(column_counts.keys())
                    n_valid = len(valid_columns)
                    
                    if n_valid > 0:
                        plt.figure(figsize=(15, 3 * ((n_valid + 1) // 2)))
                        
                        for i, col in enumerate(valid_columns):
                            plt.subplot(((n_valid + 1) // 2), 2, i+1)
                            
                            # Get the counts
                            counts = column_counts[col]
                            
                            # Format variant name
                            variant_name = col.replace('step3_output_int_', '').replace('step3_output_int', 'standard')
                            display_name = variant_name_mapping.get(variant_name, variant_name)
                            
                            # Check if this is the allow_0 variant
                            if 'allow_0' in col and 0 in counts.index:
                                # Include 0 in the range
                                plt.bar(counts.index, counts.values)
                                plt.title(display_name)
                                plt.xlabel('Predicted Sentence Number')
                                plt.ylabel('Count')
                                plt.xticks(range(0, 6))
                                plt.grid(axis='y', alpha=0.3)
                            else:
                                # Regular range 1-5
                                plt.bar(counts.index, counts.values)
                                plt.title(display_name)
                                plt.xlabel('Predicted Sentence Number')
                                plt.ylabel('Count')
                                plt.xticks(range(1, 6))
                                plt.grid(axis='y', alpha=0.3)
                            
                            # Add count labels
                            for x, y in zip(counts.index, counts.values):
                                plt.text(x, y + 0.1, str(y), ha='center')
                        
                        plt.tight_layout()
                        hist_plot_path = os.path.join(VARIANTS_PATH, 'predicted_sentence_histograms.png')
                        plt.savefig(hist_plot_path, dpi=300)
                        print(f"Saved histograms to: {hist_plot_path}")
                    
                    # Create a combined histogram
                    if n_valid > 0:
                        plt.figure(figsize=(10, 6))
                        
                        # For each variant, create a bar in a grouped bar chart
                        bar_width = 0.15
                        sentence_nums = [1, 2, 3, 4, 5]
                        positions = np.array(range(len(sentence_nums)))
                        
                        # Set different colors for each variant
                        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                        
                        for i, col in enumerate(valid_columns):
                            # Get counts for each sentence number
                            counts = column_counts[col]
                            
                            # Format variant name
                            variant_name = col.replace('step3_output_int_', '').replace('step3_output_int', 'standard')
                            display_name = variant_name_mapping.get(variant_name, variant_name)
                            
                            # Initialize counts dictionary with zeros for sentences 1-5
                            counts_dict = {num: 0 for num in sentence_nums}
                            
                            # Update with actual counts
                            for num in sentence_nums:
                                if num in counts.index:
                                    counts_dict[num] = counts[num]
                            
                            # Convert to list in order
                            count_values = [counts_dict[num] for num in sentence_nums]
                            
                            # Plot bars with offset
                            plt.bar(
                                positions + (i - len(valid_columns)/2 + 0.5) * bar_width, 
                                count_values,
                                bar_width,
                                label=display_name,
                                color=colors[i % len(colors)]
                            )
                        
                        # Add labels and legend
                        plt.xlabel('Predicted Sentence Number')
                        plt.ylabel('Count')
                        plt.title('Distribution of Predicted Sentence Numbers by Variant')
                        plt.xticks(positions, sentence_nums)
                        plt.legend(loc='best')
                        plt.grid(axis='y', alpha=0.3)
                        plt.tight_layout()
                        
                        combined_hist_path = os.path.join(HISTOGRAM_PATH, 'combined_sentence_histogram.png')
                        plt.savefig(combined_hist_path, dpi=300)
                        print(f"Saved combined histogram to: {combined_hist_path}")
                
                except Exception as e:
                    print(f"Error creating histograms: {e}")
                    import traceback
                    traceback.print_exc()
                
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()

def generate_placeholder_figures(summaries):
    """
    Generate placeholder figures based on summary data when actual data isn't available.
    
    Args:
        summaries: Dictionary with variant summaries
    """
    if not PLOTTING_AVAILABLE:
        print("Cannot generate placeholder figures: plotting libraries not available")
        return
    
    try:
        # Create accuracy bar chart based on summary data
        plt.figure(figsize=(10, 6))
        
        # Extract variant names and accuracies
        variants = []
        accuracies = []
        
        for variant, data in summaries.items():
            if 'accuracy' in data and not isinstance(data.get('placeholder', False), bool):
                variants.append(variant.replace('alternative_m3_', ''))
                accuracies.append(data['accuracy'] * 100)  # Convert to percentage
        
        if not variants:
            print("No valid accuracy data in summaries")
            return
            
        # Create bar chart
        plt.bar(variants, accuracies, color='skyblue')
        plt.xlabel('Variant')
        plt.ylabel('Accuracy (%)')
        plt.title('Sentence Recognition Accuracy by Variant (Placeholder)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Add percentage labels
        for i, v in enumerate(accuracies):
            plt.text(i, v + 1, f"{v:.1f}%", ha='center')
        
        # Save the placeholder figure
        placeholder_path = os.path.join(VARIANTS_PATH, 'placeholder_accuracy.png')
        plt.savefig(placeholder_path, dpi=300)
        print(f"Generated placeholder accuracy chart: {placeholder_path}")
        
    except Exception as e:
        print(f"Error generating placeholder figures: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Main function to read the merged data and display column names,
    then merge it with the processed output file.
    """
    print(f"\nStarting data analysis process...")
    print(f"Merged variants file: {MERGED_FILE}")
    print(f"Output file to merge with: {OUTPUT_FILE}")
    
    # Check if variant summaries exist (to determine if we're using real data or placeholders)
    variant_summaries_exist = check_variant_summaries()
    print("\nVariant summary status:")
    for variant, exists in variant_summaries_exist.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {variant}")
    
    # If we have summaries, load them
    variant_summaries = load_variant_summaries()
    
    # Check if merged file exists
    if not os.path.exists(MERGED_FILE):
        print(f"Error: Merged file {MERGED_FILE} does not exist.")
        
        # Check if output directories exist
        merged_dir = os.path.dirname(MERGED_FILE)
        if not os.path.exists(merged_dir):
            print(f"  Directory {merged_dir} does not exist. Create it with: mkdir -p {merged_dir}")
        else:
            # List files in the directory
            files = os.listdir(merged_dir)
            if files:
                print(f"  Files in {merged_dir}:")
                for file in files:
                    print(f"    {file}")
            else:
                print(f"  Directory {merged_dir} exists but is empty.")
        
        # Generate placeholder figures if we have summaries
        if variant_summaries and PLOTTING_AVAILABLE:
            print("\nGenerating placeholder figures based on summary data...")
            generate_placeholder_figures(variant_summaries)
            
        return
    
    # Check if output file exists - this is optional, we can still analyze the merged file
    use_output_file = True
    if not os.path.exists(OUTPUT_FILE):
        print(f"Warning: Output file {OUTPUT_FILE} does not exist. Proceeding with only merged data.")
        use_output_file = False
    
    # Read the merged data
    try:
        print(f"\nReading merged variants data...")
        merged_df = pd.read_csv(MERGED_FILE, sep='|')
        print(f"Successfully loaded merged data with {len(merged_df)} rows")
        
        # Check if the merged data is empty (placeholder)
        if merged_df.empty:
            print("Warning: Merged data is empty (likely a placeholder).")
            
            # Generate placeholder figures if we have summaries
            if variant_summaries and PLOTTING_AVAILABLE:
                print("\nGenerating placeholder figures based on summary data...")
                generate_placeholder_figures(variant_summaries)
                
            return
        
        # Display column names of the merged data
        print(f"\nColumns in the merged data ({len(merged_df.columns)} columns):")
        for i, col in enumerate(merged_df.columns):
            print(f"{i+1}. {col}")
            
        # Count non-null values in each column
        print(f"\nColumn statistics (non-null values):")
        for col in merged_df.columns:
            non_null_count = merged_df[col].count()
            percentage = non_null_count / len(merged_df) * 100
            print(f"  {col}: {non_null_count}/{len(merged_df)} ({percentage:.1f}%)")
        
        # If output file exists, read it and merge
        if use_output_file:
            print(f"\nReading output file...")
            output_df = pd.read_csv(OUTPUT_FILE, sep='|')
            print(f"Successfully loaded output data with {len(output_df)} rows")
            
            # Display column names of the output file
            print(f"\nColumns in the output file ({len(output_df.columns)} columns):")
            for i, col in enumerate(output_df.columns):
                print(f"{i+1}. {col}")
            
            # Check if key column exists in both dataframes
            key_column = "step2_output_nth_sentence_message_only"
            
            if key_column not in merged_df.columns:
                print(f"Error: Key column '{key_column}' not found in merged data.")
                return
                
            if key_column not in output_df.columns:
                print(f"Error: Key column '{key_column}' not found in output data.")
                return
                
            # Check key column values
            print(f"\nKey column '{key_column}' statistics:")
            print(f"  Unique values in merged data: {merged_df[key_column].nunique()}")
            print(f"  Unique values in output data: {output_df[key_column].nunique()}")
            
            # Perform the merge
            print(f"\nMerging dataframes on key column '{key_column}'...")
            final_df = pd.merge(
                merged_df,
                output_df,
                on=key_column,
                how='outer'
            )
            
            # Display merged result stats
            print(f"Merged result has {len(final_df)} rows and {len(final_df.columns)} columns")
            print(f"  From merged data: {len(merged_df)} rows")
            print(f"  From output data: {len(output_df)} rows")
            print(f"  Rows only in merged data: {len(final_df) - len(output_df.merge(final_df, on=key_column))}")
            print(f"  Rows only in output data: {len(final_df) - len(merged_df.merge(final_df, on=key_column))}")
        else:
            # No output file, just use merged data
            final_df = merged_df
            print("\nUsing only merged data for analysis (no output file)")
        
        # Display a sample of columns
        print(f"\nColumns in the final dataset (showing first 30):")
        for i, col in enumerate(final_df.columns[:30]):
            print(f"{i+1}. {col}")
            
        if len(final_df.columns) > 30:
            print(f"...and {len(final_df.columns) - 30} more columns")
        
        # Save the merged result
        os.makedirs(os.path.dirname(FINAL_OUTPUT_FILE), exist_ok=True)
        final_df.to_csv(FINAL_OUTPUT_FILE, index=False, sep='|')
        print(f"\nFinal merged data saved to: {FINAL_OUTPUT_FILE}")
        
        # Convert output columns to integers before analysis
        # List of output columns to analyze
        output_columns = [
            'step3_output_int',
            'step3_output_int_alternative_full_sentence',
            'step3_output_int_alternative_cot',
            'step3_output_int_alternative_allow_0',
            'step3_output_int_alternative_m1_unchanged',
            'step3_output_int_alternative_numbered_sentences',
            'step3_output_int_alternative_revealed_recognition_task'
        ]

        # Convert reference column to integer as well
        reference_column = 'step2_random_sent_num'
        if reference_column in final_df.columns:
            # Convert to numeric first (in case there are non-numeric values)
            final_df[reference_column] = pd.to_numeric(final_df[reference_column], errors='coerce')
            # Then convert to integer (NaN values will remain NaN)
            final_df[reference_column] = final_df[reference_column].astype('Int64')
            print(f"Converted reference column '{reference_column}' to integer type")
        else:
            print(f"Warning: Reference column '{reference_column}' not found in the dataset")
        
        # Convert each output column to integer
        for col in output_columns:
            if col in final_df.columns:
                # Convert to numeric first (in case there are non-numeric values)
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
                # Then convert to integer (NaN values will remain NaN)
                final_df[col] = final_df[col].astype('Int64')
                print(f"Converted column '{col}' to integer type")
            else:
                print(f"Warning: Column '{col}' not found in the dataset")
        
        # If reference column is missing, we can't calculate accuracy
        if reference_column not in final_df.columns:
            print("\nCannot calculate accuracy - reference column missing.")
            return
        
        # Calculate accuracy for each output column
        print(f"\n{'-'*80}")
        print(f"ACCURACY ANALYSIS: Comparing predicted sentence numbers with actual sentence numbers")
        print(f"{'-'*80}")
        
        # Create a results table
        accuracy_results = []
        
        # Data for plotting
        plot_data = []
        
        # Store raw column counts for histograms
        column_counts = {}
        
        for col in output_columns:
            if col in final_df.columns:
                # Create a mask for valid rows (both the output column and reference column have values)
                valid_mask = (~final_df[col].isna()) & (~final_df[reference_column].isna())
                
                # Count valid rows
                valid_count = valid_mask.sum()
                
                if valid_count > 0:
                    # Calculate accuracy - explicitly convert to integers for comparison
                    ref_values = final_df.loc[valid_mask, reference_column].fillna(-1).astype(int)
                    col_values = final_df.loc[valid_mask, col].fillna(-1).astype(int)
                    
                    matches = (col_values == ref_values).sum()
                    accuracy = matches / valid_count * 100
                    
                    # Count distribution of predicted numbers and store for histogram
                    value_counts = col_values.value_counts().sort_index()
                    
                    # Filter out the -1 values (which were NaN)
                    if -1 in value_counts.index:
                        value_counts = value_counts.drop(-1)
                    
                    value_distribution = ', '.join([f"{int(num)}: {count}" for num, count in value_counts.items()])
                    
                    # Store counts for histogram plotting
                    column_counts[col] = value_counts
                    
                    # Add to results
                    accuracy_results.append({
                        'Column': col,
                        'Valid Samples': valid_count,
                        'Correct Matches': matches,
                        'Accuracy (%)': accuracy,
                        'Distribution': value_distribution
                    })
                    
                    # Prepare data for plotting
                    # Calculate distribution by actual sentence number
                    for actual_sent_num in range(1, 6):
                        # Use integer comparison
                        sent_mask = valid_mask & (final_df[reference_column].fillna(-1).astype(int) == actual_sent_num)
                        if sent_mask.sum() > 0:
                            # Calculate percentage distribution for this sentence number
                            for pred_num in range(1, 6):
                                # Use integer comparison
                                pred_count = (final_df.loc[sent_mask, col].fillna(-1).astype(int) == pred_num).sum()
                                percentage = pred_count / sent_mask.sum() * 100
                                
                                # Add to plot data
                                variant_name = col.replace('step3_output_int_', '').replace('step3_output_int', 'standard')
                                plot_data.append({
                                    'actual_sent_num': int(actual_sent_num),
                                    'predicted_sent_num': int(pred_num),
                                    'percentage': percentage,
                                    'variant': variant_name
                                })
                else:
                    print(f"Warning: No valid data to calculate accuracy for column '{col}'")
            else:
                print(f"Warning: Column '{col}' not found in the merged dataset")
        
        # Display results as a table
        if accuracy_results:
            # Create a results table
            results_df = pd.DataFrame(accuracy_results)
            
            # Print the summary table
            print("\nAccuracy Results:")
            print(results_df[['Column', 'Valid Samples', 'Correct Matches', 'Accuracy (%)']].to_string(index=False))
            
            # Print distribution details
            print("\nDistribution of Predicted Sentence Numbers:")
            for result in accuracy_results:
                print(f"{result['Column']}:")
                print(f"  {result['Distribution']}")
                print()
        else:
            print("No accuracy results could be calculated")
            
        # Create visualizations if we have plot data and plotting libraries
        if plot_data and PLOTTING_AVAILABLE:
            try:
                # Convert to dataframe for plotting
                plot_df = pd.DataFrame(plot_data)
                
                # Create visualizations using only the specified paths
                print(f"\nGenerating plots in {VARIANTS_PATH} and {HISTOGRAM_PATH}...")
                
                # Create consolidated plot data for both visualizations
                # Calculate percentages for all variants
                percentage_data = []

                # Calculate accuracy for each sentence number and variant
                accuracy_by_sentence = []

                # Extract accuracy for each sentence position across all variants
                for col in output_columns:
                    if col in final_df.columns:
                        # Create a mask for valid rows (both the output column and reference column have values)
                        valid_mask = (~final_df[col].isna()) & (~final_df[reference_column].isna())
                        
                        if valid_mask.sum() > 0:
                            # Get variant name
                            variant_name = col.replace('step3_output_int_', '').replace('step3_output_int', 'standard')
                            
                            # For each sentence number
                            for sent_num in range(1, 6):
                                # Filter for actual == sent_num
                                sent_mask = valid_mask & (final_df[reference_column].fillna(-1).astype(int) == sent_num)
                                
                                if sent_mask.sum() > 0:
                                    # Calculate accuracy for this sentence position
                                    correct_preds = (final_df.loc[sent_mask, col].fillna(-1).astype(int) == sent_num).sum()
                                    accuracy = (correct_preds / sent_mask.sum()) * 100
                                    
                                    # Add to accuracy_by_sentence
                                    accuracy_by_sentence.append({
                                        'variant': variant_name,
                                        'sentence_number': sent_num,
                                        'accuracy': accuracy,
                                        'sample_size': sent_mask.sum()
                                    })

                # Generate visualizations if we have accuracy data
                create_accuracy_visualizations(accuracy_by_sentence)
                
                # Create distribution visualizations from column counts 
                create_distribution_visualizations(column_counts, accuracy_results)
                
            except Exception as e:
                print(f"\nError creating plots: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"Error during data processing: {e}")
        import traceback
        traceback.print_exc()
        
        # Generate placeholder figures if we have summaries
        if variant_summaries and PLOTTING_AVAILABLE:
            print("\nGenerating placeholder figures based on summary data...")
            generate_placeholder_figures(variant_summaries)

if __name__ == "__main__":
    main()