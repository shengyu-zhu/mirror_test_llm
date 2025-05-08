import os
import pandas as pd
import numpy as np
from plotnine import *
from tqdm import tqdm

def main():
    # Paths
    OUTPUT_PATH = "./output/"
    FIGURE_PATH = os.path.join(OUTPUT_PATH, "figure/sentence_num_distribution_predicted/")
    os.makedirs(FIGURE_PATH, exist_ok=True)

    # Use individual files approach similar to the second script
    INPUT_DATA_PATH = "./data/step3/output_processed/"
    OUTPUT_FILE_TEMPLATE = f"{INPUT_DATA_PATH}mirror_test_results_{{}}_{{}}.csv"
    MODELS = ["ChatGPT", "Claude", "Grok", "Gemini", "Llama", "Deepseek"]
    
    # Create an empty DataFrame to store all the data
    df = pd.DataFrame()
    
    # Load data from individual files for each model pair
    print("Loading data from individual files...")
    for model1 in MODELS:
        for model2 in MODELS:
            file_path = OUTPUT_FILE_TEMPLATE.format(model1, model2)
            try:
                model_df = pd.read_csv(file_path, sep="|", engine="python")
                # Add model information
                model_df['model1'] = model1
                model_df['model2'] = model2
                # Append to the combined dataframe
                df = pd.concat([df, model_df], ignore_index=True)
                print(f"Loaded data from {file_path}")
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
    
    print(f"Successfully loaded {len(df)} records from individual files.")
    
    # Rename for consistency
    df['Model_1'] = df['model1']
    df['Model_2'] = df['model2']
    
    # Clean and filter data
    print("Cleaning and filtering data...")
    df['step2_random_sent_num'] = pd.to_numeric(df['step2_random_sent_num'], errors='coerce')
    df['step3_output_int'] = pd.to_numeric(df['step3_output_int'], errors='coerce')
    
    # Filter to valid data
    df = df[df['step2_random_sent_num'].isin([1, 2, 3, 4, 5])]
    
    # Define the distribution computation function for standard output
    def compute_distribution(series, label):
        valid_series = series[series.isin([1, 2, 3, 4, 5])]
        if valid_series.empty:
            return pd.DataFrame(columns=['step_num', 'percentage', 'source'])
            
        pct = (valid_series.value_counts(normalize=True) * 100).reset_index()
        pct.columns = ['step_num', 'percentage']
        pct['step_num'] = pct['step_num'].astype(int)
        pct['source'] = label
        return pct

    # Calculate percentage distributions with updated labels for standard data
    step2_dist = compute_distribution(df['step2_random_sent_num'], '# of actual revised sentence')
    step3_dist = compute_distribution(df['step3_output_int'], '# of "strange" sentence')
    
    # Calculate accuracy per step2_random_sent_num with updated label for standard data
    standard_accuracy_data = df.copy()
    standard_accuracy_data['correct'] = standard_accuracy_data['step2_random_sent_num'] == standard_accuracy_data['step3_output_int']
    standard_accuracy_df = (
        standard_accuracy_data.groupby('step2_random_sent_num')['correct']
        .mean()
        .reset_index()
        .rename(columns={'correct': 'percentage'})
    )
    standard_accuracy_df['percentage'] *= 100
    standard_accuracy_df['step_num'] = standard_accuracy_df['step2_random_sent_num'].astype(int)
    standard_accuracy_df['source'] = 'accuracy'
    standard_accuracy_df = standard_accuracy_df[['step_num', 'percentage', 'source']]

    # Model-specific distributions for step3_output_int (standard)
    model1_dists = []
    model2_dists = []
    
    print("Computing model-specific distributions for standard output...")
    for model in MODELS:
        subset = df[df['Model_1'] == model]
        if not subset.empty and subset['step3_output_int'].notna().any():
            valid_subset = subset[subset['step3_output_int'].isin([1, 2, 3, 4, 5])]
            if not valid_subset.empty:
                dist = compute_distribution(valid_subset['step3_output_int'], f'Model_1:{model}')
                model1_dists.append(dist)

    for model in MODELS:
        subset = df[df['Model_2'] == model]
        if not subset.empty and subset['step3_output_int'].notna().any():
            valid_subset = subset[subset['step3_output_int'].isin([1, 2, 3, 4, 5])]
            if not valid_subset.empty:
                dist = compute_distribution(valid_subset['step3_output_int'], f'Model_2:{model}')
                model2_dists.append(dist)

    model1_df = pd.concat(model1_dists, ignore_index=True) if model1_dists else pd.DataFrame()
    model2_df = pd.concat(model2_dists, ignore_index=True) if model2_dists else pd.DataFrame()

    print("Generating plots...")

    # Plot 1: Combined standard distributions
    summary_df = pd.concat([
        step2_dist, 
        step3_dist,
        standard_accuracy_df
    ]).reset_index(drop=True)
    
    # Filter out empty rows
    summary_df = summary_df[summary_df['step_num'].notna()]
    
    # If we have data, create the plot
    if not summary_df.empty:
        plot1 = (
            ggplot(summary_df, aes(x='step_num', y='percentage', color='source'))
            + geom_line(size=1.2)
            + geom_point(size=3)
            + scale_x_continuous(breaks=[1, 2, 3, 4, 5])
            + scale_y_continuous(
                limits=[0, 60],
                breaks=[0, 10, 20, 30, 40, 50, 60],
                labels=lambda x: [f"{v}%" for v in x]
            )
            + scale_fill_brewer(type='qual', palette='Dark2')
            + scale_color_brewer(type='qual', palette='Dark2')
            + theme_minimal()
            + theme(
                legend_position='bottom', 
                figure_size=(6, 5),
                panel_grid_minor=element_blank(),
                panel_grid_major_y=element_line(color='lightgray'),
                panel_grid_major_x=element_blank(),
                legend_title=element_blank()
            )
            + labs(
                title='Sentence Number Distribution & Accuracy',
                x='Sentence Number (1-5)',
                y='Percentage'
            )
        )
        plot1_path = os.path.join(FIGURE_PATH, "step_number_summary.png")
        plot1.save(plot1_path, dpi=300)
        print(f"Saved: {plot1_path}")
    else:
        print("Warning: No data available for combined plot")

    # Plot 2: Model_1 Distributions (standard)
    if not model1_df.empty:
        plot2 = (
            ggplot(model1_df, aes(x='step_num', y='percentage', color='source'))
            + geom_line(size=1.2)
            + geom_point(size=3)
            + scale_x_continuous(breaks=[1, 2, 3, 4, 5])
            + scale_y_continuous(
                limits=[0, 60],
                breaks=[0, 10, 20, 30, 40, 50, 60],
                labels=lambda x: [f"{v}%" for v in x]
            )
            + scale_fill_brewer(type='qual', palette='Dark2')
            + scale_color_brewer(type='qual', palette='Dark2')
            + theme_minimal()
            + theme(
                legend_position='bottom', 
                figure_size=(6, 5),
                panel_grid_minor=element_blank(),
                panel_grid_major_y=element_line(color='lightgray'),
                panel_grid_major_x=element_blank(),
                legend_title=element_blank()
            )
            + labs(
                title='Step 3 Output Distribution by Model 1',
                x='Sentence Number (1-5)',
                y='Percentage'
            )
        )
        plot2_path = os.path.join(FIGURE_PATH, "step_number_distribution_model1.png")
        plot2.save(plot2_path, dpi=300)
        print(f"Saved: {plot2_path}")
    else:
        print("Warning: No data available for Model 1 plot")

    # Plot 3: Model_2 Distributions (standard)
    if not model2_df.empty:
        plot3 = (
            ggplot(model2_df, aes(x='step_num', y='percentage', color='source'))
            + geom_line(size=1.2)
            + geom_point(size=3)
            + scale_x_continuous(breaks=[1, 2, 3, 4, 5])
            + scale_y_continuous(
                limits=[0, 60],
                breaks=[0, 10, 20, 30, 40, 50, 60],
                labels=lambda x: [f"{v}%" for v in x]
            )
            + scale_fill_brewer(type='qual', palette='Dark2')
            + scale_color_brewer(type='qual', palette='Dark2')
            + theme_minimal()
            + theme(
                legend_position='bottom', 
                figure_size=(6, 5),
                panel_grid_minor=element_blank(),
                panel_grid_major_y=element_line(color='lightgray'),
                panel_grid_major_x=element_blank(),
                legend_title=element_blank()
            )
            + labs(
                title='Step 3 Output Distribution by Model 2',
                x='Sentence Number (1-5)',
                y='Percentage'
            )
        )
        plot3_path = os.path.join(FIGURE_PATH, "step_number_distribution_model2.png")
        plot3.save(plot3_path, dpi=300)
        print(f"Saved: {plot3_path}")
    else:
        print("Warning: No data available for Model 2 plot")

    print(f"Analysis complete. All plots saved to {FIGURE_PATH}")

if __name__ == "__main__":
    main()