#!/usr/bin/env python3
"""
Sentence Number Accuracy Analysis Script

This script analyzes how accuracy varies by sentence number (step2_random_sent_num)
for each model pair in the mirror test results.

It produces plots showing:
1. Accuracy vs sentence number for all model pairs
2. Self-recognition accuracy vs sentence number (same model comparisons)
3. Distribution of predictions (step3_output_int) by sentence number

Usage:
    python sentence_num_accuracy.py [--baseline]
"""

import os
import pandas as pd
import numpy as np
import argparse
from plotnine import *
from tqdm import tqdm

# Configuration
INPUT_DATA_PATH = "./data/step3/output_processed/"
BASELINE_DATA_PATH = "./data/step3/output_processed_m1_output_unchanged/"
OUTPUT_PATH = "./output/"
FIGURE_PATH = os.path.join(OUTPUT_PATH, "figure/")
TABLE_PATH = os.path.join(OUTPUT_PATH, "table/")
OUTPUT_FILE_TEMPLATE = os.path.join(INPUT_DATA_PATH, "mirror_test_results_{}_{}.csv")
BASELINE_OUTPUT_FILE_TEMPLATE = os.path.join(BASELINE_DATA_PATH, "mirror_test_results_{}_{}.csv")

MODELS = ["ChatGPT", "Claude", "Grok", "Gemini", "Llama", "Deepseek"]

# Ensure output directories exist
os.makedirs(FIGURE_PATH, exist_ok=True)
os.makedirs(TABLE_PATH, exist_ok=True)

def analyze_accuracy_by_sentence_number(use_baseline=False):
    """Analyze accuracy by sentence number for each model pair."""
    data_by_sent_num = []
    all_data = []
    
    file_template = BASELINE_OUTPUT_FILE_TEMPLATE if use_baseline else OUTPUT_FILE_TEMPLATE
    output_suffix = "_m1_output_unchanged" if use_baseline else ""
    
    for model1 in tqdm(MODELS, desc="Processing Model 1"):
        for model2 in tqdm(MODELS, desc=f"Processing {model1} vs", leave=False):
            file_path = file_template.format(model1, model2)
            try:
                df = pd.read_csv(file_path, sep="|")
                df['Model_1'] = model1
                df['Model_2'] = model2
                df['Is_Same_Model'] = model1 == model2
                all_data.append(df)

                for sent_num, group in df.groupby('step2_random_sent_num'):
                    matches = (group['step2_random_sent_num'] == group['step3_output_int']).sum()
                    accuracy = matches / len(group) if len(group) else 0

                    data_by_sent_num.append({
                        'Model_1': model1,
                        'Model_2': model2,
                        'step2_random_sent_num': sent_num,
                        'accuracy': accuracy,
                        'group_size': len(group),
                        'Is_Same_Model': model1 == model2
                    })
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    df_sent_accuracy = pd.DataFrame(data_by_sent_num)
    output_file = os.path.join(TABLE_PATH, f"accuracy_by_sentence_number{output_suffix}.csv")
    df_sent_accuracy.to_csv(output_file, index=False)
    print(f"Saved raw data to {output_file}")

    if all_data:
        df_all = pd.concat(all_data, ignore_index=True)
        output_file = os.path.join(TABLE_PATH, f"full_combined_data{output_suffix}.csv")
        df_all.to_csv(output_file, index=False)
    else:
        df_all = pd.DataFrame()

    return df_sent_accuracy, df_all

def plot_accuracy_by_sent_num(df, use_baseline=False):
    """Generate plots of accuracy by sentence number."""
    output_suffix = "_m1_output_unchanged" if use_baseline else ""
    
    df['step2_random_sent_num'] = pd.to_numeric(df['step2_random_sent_num'], errors='coerce')
    df = df.dropna(subset=['step2_random_sent_num'])

    subsets = [
        (True, 'Model_1'),
        (True, 'Model_2'),
        (False, 'Model_1'),
        (False, 'Model_2')
    ]

    for is_same, group_by in subsets:
        subset_df = df[df['Is_Same_Model'] == is_same].copy()
        if subset_df.empty:
            continue

        if not is_same:
            subset_df = (subset_df
                         .groupby([group_by, 'step2_random_sent_num'], as_index=False)
                         .agg({'accuracy': 'mean'}))
            subset_df['group'] = subset_df[group_by]
        else:
            subset_df['group'] = subset_df[group_by]

        plot = (
            ggplot(subset_df, aes(x='step2_random_sent_num', y='accuracy', color='group', group='group'))
            + geom_line(size=1)
            + geom_point(size=2, alpha=0.7)
            + scale_y_continuous(limits=[0, 1.05], breaks=np.arange(0, 1.1, 0.1))
            + theme_minimal()
            + labs(
                title=f"Accuracy by Sentence Number | Is_Same_Model={is_same} grouped by {group_by}",
                x="Sentence Number",
                y="Accuracy",
                color=group_by
            )
        )
        filename = f"accuracy_by_sentence_number_is_same_{is_same}_groupby_{group_by.lower()}{output_suffix}.png"
        # Explicitly set both width and height without figure_size in theme
        plot.save(os.path.join(FIGURE_PATH, filename), dpi=300, width=5, height=4, verbose=False)
        print(f"Saved plot to {FIGURE_PATH}{filename}")

def plot_output_distribution_bar(df_all, use_baseline=False):
    """Plot distribution of step3_output_int per sentence number."""
    output_suffix = "_m1_output_unchanged" if use_baseline else ""
    
    if df_all.empty:
        print("No data available for bar plot.")
        return

    df_all['step2_random_sent_num'] = pd.to_numeric(df_all['step2_random_sent_num'], errors='coerce')
    df_all['step3_output_int'] = pd.to_numeric(df_all['step3_output_int'], errors='coerce')
    df_all = df_all.dropna(subset=['step2_random_sent_num', 'step3_output_int'])

    count_df = df_all.groupby(['step2_random_sent_num', 'step3_output_int']).size().reset_index(name='count')
    total_df = count_df.groupby('step2_random_sent_num')['count'].transform('sum')
    count_df['percentage'] = 100 * count_df['count'] / total_df

    bar_plot = (
        ggplot(count_df, aes(x='factor(step2_random_sent_num)', y='percentage', fill='factor(step3_output_int)'))
        + geom_bar(stat='identity', position='stack')
        + theme_minimal()
        + theme(legend_position='right')
        + labs(
            title='Distribution of step3_output_int by Sentence Number',
            x='Sentence Number',
            y='Percentage',
            fill='step3_output_int'
        )
    )
    filename = f"step3_output_distribution_by_sentence_number{output_suffix}.png"
    # Explicitly setting dimensions and disabling verbose output to hide size warnings
    bar_plot.save(os.path.join(FIGURE_PATH, filename), dpi=300, width=5, height=4, verbose=False)
    print(f"Saved bar plot to {FIGURE_PATH}{filename}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze mirror test results by sentence number')
    parser.add_argument('--baseline', action='store_true', help='Use baseline data from _m1_output_unchanged directory')
    args = parser.parse_args()
    
    data_source = "baseline (_m1_output_unchanged)" if args.baseline else "standard"
    
    print(f"\n=== Running Sentence Number Accuracy Analysis using {data_source} data ===\n")
    print("Step 1: Analyzing accuracy by sentence number...")
    df_sent_accuracy, df_all = analyze_accuracy_by_sentence_number(args.baseline)
    print("\nStep 2: Creating visualizations...")
    plot_accuracy_by_sent_num(df_sent_accuracy, args.baseline)
    plot_output_distribution_bar(df_all, args.baseline)
    print("\n=== Analysis Complete ===")
    print(f"Results saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()