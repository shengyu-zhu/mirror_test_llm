import pandas as pd
import numpy as np
from plotnine import *
import os
from typing import List, Dict, Tuple, Optional
import re

# Configuration
BASE_DATA_PATH = "./data/"  # Adjusted to match the path in the data collection script
OUTPUT_PATH = "./output/"
FIGURE_PATH = os.path.join(OUTPUT_PATH, "figure/effect_of_temperature/")
TABLE_PATH = f"{OUTPUT_PATH}table/"

# Create output directories if they don't exist
os.makedirs(FIGURE_PATH, exist_ok=True)
os.makedirs(TABLE_PATH, exist_ok=True)

def add_confidence_interval(df: pd.DataFrame, n_samples: int, confidence: float = 0.95) -> pd.DataFrame:
    """
    Add confidence interval to recognition accuracy results.
    
    Args:
        df: DataFrame with recognition accuracy results
        n_samples: Number of samples used for each test
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        DataFrame with added confidence interval columns
    """
    from scipy import stats
    
    # Calculate confidence intervals
    df_with_ci = df.copy()
    
    for idx, row in df.iterrows():
        # Get accuracy value
        p = row['Recognition_accuracy']
        
        # Calculate standard error and confidence interval
        # Using Wilson score interval for proportion confidence interval
        if 0 < p < 1:  # Wilson interval undefined at p=0 or p=1
            z = stats.norm.ppf(1 - (1 - confidence) / 2)
            n = n_samples
            
            # Wilson score interval
            denominator = 1 + z**2/n
            centre_adjusted_probability = (p + z**2/(2*n))/denominator
            adjusted_standard_deviation = np.sqrt((p*(1-p) + z**2/(4*n))/n)/denominator
            
            lower_bound = centre_adjusted_probability - z * adjusted_standard_deviation
            upper_bound = centre_adjusted_probability + z * adjusted_standard_deviation
        else:
            # Use normal approximation at boundaries with a small offset
            p_adj = max(0.001, min(0.999, p))
            se = np.sqrt(p_adj * (1 - p_adj) / n_samples)
            z_value = stats.norm.ppf(1 - (1 - confidence) / 2)
            margin = z_value * se
            
            lower_bound = max(0, p - margin)
            upper_bound = min(1, p + margin)
        
        # Add CI to dataframe
        df_with_ci.loc[idx, 'ci_lower_bound'] = round(lower_bound, 3)
        df_with_ci.loc[idx, 'ci_upper_bound'] = round(upper_bound, 3)
        
    return df_with_ci


def analyze_temperature_impact(models: List[str] = None) -> pd.DataFrame:
    """
    Analyze impact of temperature parameters on recognition accuracy.
    
    Args:
        models: List of models to analyze. Default is ["Gemini", "Claude"]
        
    Returns:
        DataFrame with temperature impact analysis results
    """
    if models is None:
        # Default models for temperature analysis - using model names from the data generation script
        model1 = "Gemini"
        model2 = "Claude"
    else:
        # Fixed: Handle when models is a list of any length
        if len(models) >= 2:
            model1, model2 = models[0], models[1]
        else:
            print("Error: models list must contain at least 2 elements")
            return pd.DataFrame()
        
    print(f"\nAnalyzing temperature impact for {model1} vs {model2}")
    
    data_acc = []
    parameter_space = np.round(np.linspace(0, 1, 3).tolist(), 2)
    
    # Process each temperature combination
    for m1_temperature_value in parameter_space:
        for m2_temperature_value in parameter_space:
            # Load data
            file_path = f"{BASE_DATA_PATH}temperature_result/mirror_test_results_{model1}_{model2}_{m1_temperature_value}_{m2_temperature_value}.csv"
            try:
                df = pd.read_csv(file_path)
                
                # Check if the file contains valid data
                if 'step2_output' not in df.columns or 'step3_output_message_only' not in df.columns:
                    print(f"Warning: Missing required columns in {file_path}")
                    continue
                
                # Fixed: Define valid_df and prediction_col variables
                valid_df = df.dropna(subset=['step2_output', 'step3_output_message_only'])
                prediction_col = 'step3_output_int'  # Assuming this column exists or needs to be created
                
                # Fixed: Added proper column creation for step3_output_int if it doesn't exist
                if prediction_col not in valid_df.columns:
                    # Extract the prediction from step3_output_message_only if needed
                    # This is a placeholder - adjust based on actual data format
                    # First check if step3_output_message_only contains string values
                    if valid_df['step3_output_message_only'].dtype == 'object':
                        # Only use str accessor if we have string values
                        try:
                            valid_df[prediction_col] = pd.to_numeric(
                                valid_df['step3_output_message_only'].astype(str).str.extract(r'(\d+)').iloc[:, 0], 
                                errors='coerce'
                            ).fillna(0).astype(int)
                        except Exception as e:
                            print(f"Warning: Could not extract numbers from step3_output_message_only: {e}")
                            # Fallback: If extraction fails, set to default value
                            valid_df[prediction_col] = 0
                    else:
                        # If not strings, convert directly
                        valid_df[prediction_col] = pd.to_numeric(
                            valid_df['step3_output_message_only'], 
                            errors='coerce'
                        ).fillna(0).astype(int)
                
                valid_df['prediction_int'] = pd.to_numeric(
                    valid_df[prediction_col], 
                    errors='coerce'
                ).fillna(0).astype(int)
                
                # Calculate accuracy
                accuracy = sum(valid_df['prediction_int'] == valid_df['step2_random_sent_num']) / len(valid_df)
                valid_df['is_correct'] = (valid_df['prediction_int'] == valid_df['step2_random_sent_num'])
        
                # Store results
                data_acc.append({
                    'Model_1': model1,
                    'Model_2': model2,
                    'm1_temperature_value': m1_temperature_value,
                    'm2_temperature_value': m2_temperature_value,
                    'Recognition_accuracy': round(accuracy, 3),
                    'Valid_samples': len(valid_df)
                })
                
                print(f"Temperature {m1_temperature_value}/{m2_temperature_value}: Accuracy = {accuracy:.3f}")
                
            except FileNotFoundError:
                print(f"File not found: {file_path} - this temperature combination may not have been processed")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    # Create DataFrame and add confidence intervals
    if not data_acc:
        print("No valid data found for temperature impact analysis")
        return pd.DataFrame()
        
    df_impact = pd.DataFrame(data_acc)
    
    # Use the actual number of valid samples for each combination
    df_impact_with_ci = df_impact.copy()
    for idx, row in df_impact.iterrows():
        n_samples = row['Valid_samples']
        if n_samples > 0:
            # Fixed: Process one row at a time for confidence interval calculation
            temp_df = pd.DataFrame([row])
            temp_df_with_ci = add_confidence_interval(temp_df, n_samples)
            df_impact_with_ci.loc[idx, 'ci_lower_bound'] = temp_df_with_ci.loc[temp_df_with_ci.index[0], 'ci_lower_bound'] 
            df_impact_with_ci.loc[idx, 'ci_upper_bound'] = temp_df_with_ci.loc[temp_df_with_ci.index[0], 'ci_upper_bound']
    
    # Save raw data
    df_impact_with_ci.to_csv(f"{TABLE_PATH}temperature_impact.csv", index=False)
    
    return df_impact_with_ci


def create_temperature_visualization(df_impact: pd.DataFrame):
    """
    Create visualization for temperature impact analysis.
    
    Args:
        df_impact: DataFrame with temperature impact data
    """
    if df_impact.empty:
        print("Empty dataframe - cannot create visualization")
        return
        
    # Convert temperature to string for plotting
    df_plot = df_impact.copy()
    df_plot['m2_temperature_value'] = df_plot['m2_temperature_value'].astype(str)
    
    # Create temperature effect heatmap for more clear visualization
    # Pivot data for heatmap
    pivot_df = df_plot.pivot(
        index='m1_temperature_value', 
        columns='m2_temperature_value', 
        values='Recognition_accuracy'
    ).reset_index()
    
    # Melt the pivot table for ggplot
    melt_df = pd.melt(
        pivot_df, 
        id_vars=['m1_temperature_value'], 
        value_vars=['0.0', '0.5', '1.0'],
        var_name='m2_temperature_value',
        value_name='Recognition_accuracy'
    )
    
    # Create temperature heatmap with updated styling
    heat_plot = (ggplot(melt_df, 
                aes(x='m1_temperature_value',
                    y='m2_temperature_value',
                    fill='Recognition_accuracy'))
            + geom_tile(aes(width=0.9, height=0.9))
            + scale_fill_gradient2(
                low="#f8766d", 
                mid="#ffffff", 
                high="#00ba38", 
                midpoint=0.25,
                limits=[0, 0.5],
                breaks=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
                labels=lambda x: [f"{int(v*100)}%" for v in x]
            )
            + theme_minimal()
            + labs(title=f'Mirror Test Accuracy: {df_impact.iloc[0]["Model_1"]} vs. {df_impact.iloc[0]["Model_2"]}',
                   x=f'{df_impact.iloc[0]["Model_1"]} Temperature',
                   y=f'{df_impact.iloc[0]["Model_2"]} Temperature',
                   fill='Recognition\nAccuracy')
            + theme(
                legend_position='bottom',
                figure_size=(1.618 * 3, 3),
                panel_grid_minor=element_blank(),
                panel_grid_major_y=element_line(color='lightgray'),
                panel_grid_major_x=element_blank()
            )
           )
    
    heat_plot.save(f"{FIGURE_PATH}temperature_heatmap.png", dpi=600)
    print(f"Temperature heatmap saved to {FIGURE_PATH}temperature_heatmap.png")
    
    # Create line plot with updated styling
    line_plot = (ggplot(df_plot, 
                aes(x='m1_temperature_value',
                    y='Recognition_accuracy',
                    color='m2_temperature_value',
                    group='m2_temperature_value'))
            + geom_line(size=1.2)  # Increased line size from 1 to 1.2
            + geom_point(size=3)
            + geom_errorbar(aes(ymin='ci_lower_bound', ymax='ci_upper_bound'), width=0.1)
            + scale_color_brewer(type='qual', palette='Dark2')  # Updated to use brewer palette
            + scale_y_continuous(
                limits=[0, 0.5],
                breaks=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
                labels=lambda x: [f"{int(v*100)}%" for v in x]
            )
            + theme_minimal()
            + labs(title=f'Mirror Test Accuracy: {df_impact.iloc[0]["Model_1"]} vs. {df_impact.iloc[0]["Model_2"]}',
                   x=f'{df_impact.iloc[0]["Model_1"]} Temperature',
                   y='Recognition Accuracy',
                   color=f'{df_impact.iloc[0]["Model_2"]}\nTemperature')
            + theme(
                legend_position='bottom',  # Changed from 'right' to 'bottom'
                figure_size=(1.618 * 3, 3),
                panel_grid_minor=element_blank(),
                panel_grid_major_y=element_line(color='lightgray'),
                panel_grid_major_x=element_blank()
            )
           )
    
    line_plot.save(f"{FIGURE_PATH}temperature_effect.png", dpi=600)
    print(f"Temperature effect plot saved to {FIGURE_PATH}temperature_effect.png")
    
    # Create line plot WITHOUT confidence intervals
    line_plot_no_ci = (ggplot(df_plot, 
                aes(x='m1_temperature_value',
                    y='Recognition_accuracy',
                    color='m2_temperature_value',
                    group='m2_temperature_value'))
            + geom_line(size=1.2)
            + geom_point(size=3)
            + scale_color_brewer(type='qual', palette='Dark2')
            + scale_y_continuous(
                limits=[0, 0.5],
                breaks=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
                labels=lambda x: [f"{int(v*100)}%" for v in x]
            )
            + theme_minimal()
            + labs(title=f'Mirror Test Accuracy: {df_impact.iloc[0]["Model_1"]} vs. {df_impact.iloc[0]["Model_2"]}',
                   x=f'{df_impact.iloc[0]["Model_1"]} Temperature',
                   y='Recognition Accuracy',
                   color=f'{df_impact.iloc[0]["Model_2"]}\nTemperature')
            + theme(
                legend_position='bottom',
                figure_size=(1.618 * 3, 3),
                panel_grid_minor=element_blank(),
                panel_grid_major_y=element_line(color='lightgray'),
                panel_grid_major_x=element_blank()
            )
           )
    
    line_plot_no_ci.save(f"{FIGURE_PATH}temperature_effect_no_ci.png", dpi=600)
    print(f"Temperature effect plot (without CI) saved to {FIGURE_PATH}temperature_effect_no_ci.png")


def run_temperature_analysis(models: List[str] = None) -> Optional[pd.DataFrame]:
    """
    Run the complete temperature impact analysis.
    
    Args:
        models: List of models to analyze
    
    Returns:
        DataFrame with temperature impact results
    """
    try:
        df_impact = analyze_temperature_impact(models)
        if not df_impact.empty:
            create_temperature_visualization(df_impact)
            
            # Create a text summary of results
            with open(f"{OUTPUT_PATH}temperature_analysis_summary.txt", "w") as f:
                f.write(f"TEMPERATURE IMPACT ANALYSIS SUMMARY\n")
                f.write(f"=================================\n\n")
                f.write(f"Models: {df_impact.iloc[0]['Model_1']} vs {df_impact.iloc[0]['Model_2']}\n\n")
                
                # Find best and worst combinations
                best_idx = df_impact['Recognition_accuracy'].idxmax()
                worst_idx = df_impact['Recognition_accuracy'].idxmin()
                
                best_row = df_impact.iloc[best_idx]
                worst_row = df_impact.iloc[worst_idx]
                
                f.write(f"Best temperature combination:\n")
                f.write(f"  {best_row['Model_1']} temperature: {best_row['m1_temperature_value']}\n")
                f.write(f"  {best_row['Model_2']} temperature: {best_row['m2_temperature_value']}\n")
                f.write(f"  Recognition accuracy: {best_row['Recognition_accuracy']:.3f}\n")
                f.write(f"  95% CI: [{best_row['ci_lower_bound']:.3f}, {best_row['ci_upper_bound']:.3f}]\n\n")
                
                f.write(f"Worst temperature combination:\n")
                f.write(f"  {worst_row['Model_1']} temperature: {worst_row['m1_temperature_value']}\n")
                f.write(f"  {worst_row['Model_2']} temperature: {worst_row['m2_temperature_value']}\n")
                f.write(f"  Recognition accuracy: {worst_row['Recognition_accuracy']:.3f}\n")
                f.write(f"  95% CI: [{worst_row['ci_lower_bound']:.3f}, {worst_row['ci_upper_bound']:.3f}]\n\n")
                
                # Add overall analysis
                f.write(f"Effect of {df_impact.iloc[0]['Model_1']} temperature (averaged across {df_impact.iloc[0]['Model_2']} temperatures):\n")
                for temp in sorted(df_impact['m1_temperature_value'].unique()):
                    avg_acc = df_impact[df_impact['m1_temperature_value'] == temp]['Recognition_accuracy'].mean()
                    f.write(f"  Temperature {temp}: {avg_acc:.3f}\n")
                
                f.write(f"\nEffect of {df_impact.iloc[0]['Model_2']} temperature (averaged across {df_impact.iloc[0]['Model_1']} temperatures):\n")
                for temp in sorted(df_impact['m2_temperature_value'].unique()):
                    avg_acc = df_impact[df_impact['m2_temperature_value'] == temp]['Recognition_accuracy'].mean()
                    f.write(f"  Temperature {temp}: {avg_acc:.3f}\n")
            
            print(f"Analysis summary saved to {OUTPUT_PATH}temperature_analysis_summary.txt")
            
        return df_impact
    except Exception as e:
        import traceback
        print(f"Error in temperature analysis: {e}")
        traceback.print_exc()
        return None


def analyze_model_comparison(models_pairs: List[List[str]] = None) -> pd.DataFrame:
    """
    Compare recognition performance across different model pairs.
    
    Args:
        models_pairs: List of model pairs to compare
        
    Returns:
        DataFrame with model comparison results
    """
    if models_pairs is None:
        # Default model pairs to compare
        models_pairs = [
            ["Grok", "ChatGPT"]
        ]
        
    print("\nAnalyzing model comparison results")
    
    comparison_results = []
    
    for m1, m2 in models_pairs:
        # Use only temperature 0.0 for fair comparison
        file_path = f"{BASE_DATA_PATH}temperature_result/mirror_test_results_{m1}_{m2}_0.0_0.0.csv"
        
        try:
            df = pd.read_csv(file_path)
            
            # Fixed: Use correct validation approach
            valid_df = df.dropna(subset=['step2_output', 'step3_output_message_only'])
            
            # Fixed: Ensure step3_output_int column exists
            if 'step3_output_int' not in valid_df.columns:
                # Extract the prediction from step3_output_message_only if needed
                # First check if step3_output_message_only contains string values
                if valid_df['step3_output_message_only'].dtype == 'object':
                    # Only use str accessor if we have string values
                    try:
                        valid_df['step3_output_int'] = pd.to_numeric(
                            valid_df['step3_output_message_only'].astype(str).str.extract(r'(\d+)').iloc[:, 0], 
                            errors='coerce'
                        ).fillna(0).astype(int)
                    except Exception as e:
                        print(f"Warning: Could not extract numbers from step3_output_message_only: {e}")
                        # Fallback: If extraction fails, set to default value
                        valid_df['step3_output_int'] = 0
                else:
                    # If not strings, convert directly
                    valid_df['step3_output_int'] = pd.to_numeric(
                        valid_df['step3_output_message_only'], 
                        errors='coerce'
                    ).fillna(0).astype(int)
            
            # Calculate accuracy
            accuracy = sum(valid_df['step3_output_int'] == valid_df["step2_random_sent_num"]) / len(valid_df)
            valid_samples = len(valid_df)
            
            comparison_results.append({
                'Story_Generator': m1,
                'Reviser': m2,
                'Recognition_accuracy': round(accuracy, 3),
                'Valid_samples': valid_samples
            })
            
            print(f"{m1} (story) vs {m2} (revise): Accuracy = {accuracy:.3f}")
            
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Create DataFrame with confidence intervals
    if not comparison_results:
        print("No valid data found for model comparison")
        return pd.DataFrame()
        
    df_comparison = pd.DataFrame(comparison_results)
    
    # Add confidence intervals using each pair's actual sample count
    df_comparison_with_ci = df_comparison.copy()
    for idx, row in df_comparison.iterrows():
        n_samples = row['Valid_samples']
        if n_samples > 0:
            # Fixed: Process one row at a time for confidence interval calculation
            temp_df = pd.DataFrame([row])
            temp_df_with_ci = add_confidence_interval(temp_df, n_samples)
            df_comparison_with_ci.loc[idx, 'ci_lower_bound'] = temp_df_with_ci.loc[temp_df_with_ci.index[0], 'ci_lower_bound'] 
            df_comparison_with_ci.loc[idx, 'ci_upper_bound'] = temp_df_with_ci.loc[temp_df_with_ci.index[0], 'ci_upper_bound']
    
    # Save raw data
    df_comparison_with_ci.to_csv(f"{TABLE_PATH}model_comparison.csv", index=False)
    
    # Create visualization with updated styling
    if not df_comparison_with_ci.empty:
        comp_plot = (ggplot(df_comparison_with_ci, 
                    aes(x='Story_Generator',
                        y='Recognition_accuracy',
                        fill='Reviser'))
                + geom_bar(stat='identity', position='dodge')
                + geom_errorbar(aes(ymin='ci_lower_bound', ymax='ci_upper_bound'), 
                               width=0.2, position=position_dodge(0.9))
                + scale_fill_brewer(type='qual', palette='Dark2')  # Updated to use brewer palette
                + scale_y_continuous(
                    limits=[0, 0.5],
                    breaks=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    labels=lambda x: [f"{int(v*100)}%" for v in x]
                )
                + theme_minimal()
                + labs(title='Model Comparison: Self-Recognition Accuracy',
                       x='Story Generator Model',
                       y='Recognition Accuracy',
                       fill='Reviser Model')
                + theme(
                    legend_position='bottom',  # Changed from 'top' to 'bottom'
                    figure_size=(1.618 * 3, 3),
                    panel_grid_minor=element_blank(),
                    panel_grid_major_y=element_line(color='lightgray'),
                    panel_grid_major_x=element_blank()
                )
               )
        
        comp_plot.save(f"{FIGURE_PATH}model_comparison.png", dpi=600, width=8, height=6)
        print(f"Model comparison plot saved to {FIGURE_PATH}model_comparison.png")
    
    return df_comparison_with_ci


if __name__ == "__main__":
    # Fixed: make sure models_pairs_tested is properly formed as a list
    models_pairs_tested = ["Grok", "ChatGPT"]  # This is a list of models, not a list of pairs
    # models_pairs_tested = ["Gemini", "Claude"]
    # This allows the file to be run independently
    print("Running temperature impact analysis...")
    df_impact = run_temperature_analysis(models_pairs_tested)
    
    print("\nRunning model comparison analysis...")
    # Fixed: analyze_model_comparison expects a list of pairs
    model_pairs = [[models_pairs_tested[0], models_pairs_tested[1]]]
    df_comparison = analyze_model_comparison(model_pairs)
    
    print("\nAnalysis complete!")