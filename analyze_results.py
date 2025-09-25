import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import re
from scipy import stats

# =============================================================================
# --- 1. SETUP & CONFIGURATION ---
# =============================================================================
# --- Directories ---
OUTPUT_DIR_CSV = "output_csv"
OUTPUT_DIR_PLOTS = "output_plots"
V_G_DIR = os.path.join(OUTPUT_DIR_CSV, "v_g_functions")
os.makedirs(OUTPUT_DIR_PLOTS, exist_ok=True)

# --- Constants from your experimental design ---
NUM_REPLICATIONS = 30
SIMULATION_STEPS = 200
NUM_WEALTH_POINTS = 1000

# --- Plotting Style ---
sns.set_theme(style="whitegrid")
palette = sns.color_palette("viridis", 5) # Define a consistent color palette

# =============================================================================
# --- 2. DATA LOADING AND AGGREGATION ---
# =============================================================================
def load_and_aggregate_data(csv_dir):
    """
    Loads all CSV files, parses metadata from filenames, and aggregates them
    into two master DataFrames for model and agent data.
    """
    all_model_files = [f for f in os.listdir(csv_dir) if "model_data" in f]
    all_agent_files = [f for f in os.listdir(csv_dir) if "agent_data" in f]

    model_df_list = []
    agent_df_list = []

    # Regex to parse filenames like: run_baseline_R_1.05_rep_1_...
    # It captures the experiment name, interest rate, and replication number.
    pattern = re.compile(r"run_([a-zA-Z_]+)_R_(\d+\.\d+)_rep_(\d+)")

    print("--- Loading and parsing model data files... ---")
    for f in all_model_files:
        match = pattern.search(f)
        if match:
            run_name, rate, rep = match.groups()
            df = pd.read_csv(os.path.join(csv_dir, f))
            df = df.rename(columns={'Unnamed: 0': 'Step'})
            df['Experiment'] = run_name.rstrip('_') # Clean up trailing underscores
            df['Rate'] = float(rate)
            df['Replication'] = int(rep)
            model_df_list.append(df)

    print("--- Loading and parsing agent data files... ---")
    for f in all_agent_files:
        match = pattern.search(f)
        if match:
            run_name, rate, rep = match.groups()
            df = pd.read_csv(os.path.join(csv_dir, f))
            df = df.rename(columns={'Unnamed: 0': 'Step'})
            df['Experiment'] = run_name.rstrip('_')
            df['Rate'] = float(rate)
            df['Replication'] = int(rep)
            agent_df_list.append(df)

    if not model_df_list or not agent_df_list:
        raise FileNotFoundError("No valid data files found. Check your CSV output and filenames.")

    # Concatenate all individual dataframes into two large ones
    aggregated_model_data = pd.concat(model_df_list, ignore_index=True)
    aggregated_agent_data = pd.concat(agent_df_list, ignore_index=True)

    print(f"Successfully loaded {len(aggregated_model_data)} model data rows and {len(aggregated_agent_data)} agent data rows.")
    return aggregated_model_data, aggregated_agent_data

# =============================================================================
# --- 3. ANALYSIS & PLOTTING FUNCTIONS ---
# =============================================================================

### --- PHASE 1: SINGLE AGENT VALIDATION --- ###
def plot_phase1_validation(v_g_dir, num_wealth_points):
    """
    Generates plots for the V and g functions to validate the model's
    theoretical foundation (Cao & Werning, 2018). 
    """
    print("\n--- Generating Phase 1: VFI Validation Plots ---")
    agent_profiles = {
        "planner": {"beta": 0.97, "delta": 0.96}, "moderate": {"beta": 0.90, "delta": 0.91},
        "procrastinator": {"beta": 0.78, "delta": 0.95}, "inverse procrastinator": {"beta": 0.96, "delta": 0.85},
        "impulsive": {"beta": 0.60, "delta": 0.80}
    }
    interest_rates = [1.05, 1.12]
    wealth_grid = np.geomspace(1e-6, 1000001, num_wealth_points)

    for rate in interest_rates:
        ## --- Plot 1: Policy Functions (g) on a LOG-LOG scale  ---
        plt.figure(figsize=(14, 8))
        for name, params in agent_profiles.items():
            filepath = os.path.join(v_g_dir, f"R_{rate}_{name}_policy_function.npy")
            if os.path.exists(filepath):
                g_func = np.load(filepath)
                R_star = 1 + (1 - params['delta']) / (params['beta'] * params['delta'])
                plt.plot(wealth_grid, g_func, label=f'{name.title()} (R* ≈ {R_star:.2f})')

        plt.plot(wealth_grid, wealth_grid, 'k--', label="k' = k (Ahorro neto cero)", alpha=0.6)
        plt.title(f'Funciones de política de ahorro g(k) para R = {rate} (Escala log-log)', fontsize=16)
        plt.xlabel("Riqueza actual (k)", fontsize=12)
        plt.ylabel("Riqueza del siguiente periodo (k')", fontsize=12)
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, f"PHASE1_policy_functions_log_R_{rate}.png"))
        plt.close()
        print(f"  > Saved LOG-SCALE policy function plot for R={rate}")

        # --- Plot 2: Policy Functions (g) on a LINEAR scale ---
        plt.figure(figsize=(14, 8))
        for name, params in agent_profiles.items():
            filepath = os.path.join(v_g_dir, f"R_{rate}_{name}_policy_function.npy")
            if os.path.exists(filepath):
                g_func = np.load(filepath)
                R_star = 1 + (1 - params['delta']) / (params['beta'] * params['delta'])
                plt.plot(wealth_grid, g_func, label=f'{name.title()} (R* ≈ {R_star:.2f})')
        
        plt.plot(wealth_grid, wealth_grid, 'k--', label="k' = k (Ahorro neto cero)", alpha=0.6)
        plt.title(f'Funciones de política de ahorro g(k) para R = {rate} (Escala lineal)', fontsize=16)
        plt.xlabel("Riqueza actual (k)", fontsize=12)
        plt.ylabel("Riqueza del siguiente periodo (k')", fontsize=12)
        plt.xlim(0, 500) # Zoom in on the behavior of lower-wealth agents
        plt.ylim(0, 500)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, f"PHASE1_policy_functions_linear_R_{rate}.png"))
        plt.close()
        print(f"  > Saved LINEAR-SCALE policy function plot for R={rate}")

        # --- Plot 3: Value Functions (V) on a LOG-X scale ---
        plt.figure(figsize=(14, 8))
        for name, params in agent_profiles.items():
            filepath = os.path.join(v_g_dir, f"R_{rate}_{name}_value_function.npy")
            if os.path.exists(filepath):
                V_func = np.load(filepath)
                plt.plot(wealth_grid, V_func, label=f'{name.title()}')
        
        plt.title(f'Funciones de valor V(k) para R = {rate} (Escala log-X)', fontsize=16)
        plt.xlabel("Riqueza actual (k)", fontsize=12)
        plt.ylabel("Utilidad vitalicia V(k)", fontsize=12)
        plt.xscale('log')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, f"PHASE1_value_functions_log_R_{rate}.png"))
        plt.close()
        print(f"  > Saved LOG-X SCALE value function plot for R={rate}")

        # --- Plot 4: Value Functions (V) on a LINEAR scale (Zoomed In) ---
        plt.figure(figsize=(14, 8))
        for name, params in agent_profiles.items():
            filepath = os.path.join(v_g_dir, f"R_{rate}_{name}_value_function.npy")
            if os.path.exists(filepath):
                V_func = np.load(filepath)
                plt.plot(wealth_grid, V_func, label=f'{name.title()}')
        
        plt.title(f'Funciones de valor V(k) para R = {rate} (Escala lineal)', fontsize=16)
        plt.xlabel("Current Wealth (k)", fontsize=12)
        plt.ylabel("Lifetime Utility V(k)", fontsize=12)
        plt.xlim(0, 500) # Zoom in on the behavior of lower-wealth agents
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, f"PHASE1_value_functions_linear_R_{rate}.png"))
        plt.close()
        print(f"  > Saved LINEAR-SCALE value function plot for R={rate}")

    # --- Plot 5: Savings RATE (g(k) / R*k) ---
        plt.figure(figsize=(14, 8))
        for name, params in agent_profiles.items():
            filepath = os.path.join(v_g_dir, f"R_{rate}_{name}_policy_function.npy")
            if os.path.exists(filepath):
                g_func = np.load(filepath) # This is k'
                
                # Calculate total available resources (R*k)
                # Add a small epsilon to avoid division by zero at the lowest wealth level
                total_resources = rate * wealth_grid + 1e-9 
                
                # Calculate the savings rate
                savings_rate = g_func / total_resources
                
                R_star = 1 + (1 - params['delta']) / (params['beta'] * params['delta'])
                plt.plot(wealth_grid, savings_rate, label=f'{name.title()} (R* ≈ {R_star:.2f})')

        plt.title(f'Tasa de Ahorro ($k\' / Rk$) para R = {rate}', fontsize=16)
        plt.xlabel("Riqueza actual (k)", fontsize=12)
        plt.ylabel("Tasa de Ahorro", fontsize=12)
        plt.xscale('log') # Keep x-axis logarithmic to see the full wealth range
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, f"PHASE1_savings_rate_R_{rate}.png"))
        plt.close()
        print(f"  > Saved SAVINGS RATE plot for R={rate}")

    # --- Plot 6: Savings RATE (Linear Scale, Zoomed In) ---
        plt.figure(figsize=(14, 8))
        for name, params in agent_profiles.items():
            filepath = os.path.join(v_g_dir, f"R_{rate}_{name}_policy_function.npy")
            if os.path.exists(filepath):
                g_func = np.load(filepath) # This is k'
                
                # Calculate total available resources (R*k)
                total_resources = rate * wealth_grid + 1e-9 
                
                # Calculate the savings rate
                savings_rate = g_func / total_resources
                
                R_star = 1 + (1 - params['delta']) / (params['beta'] * params['delta'])
                plt.plot(wealth_grid, savings_rate, label=f'{name.title()} (R* ≈ {R_star:.2f})')

        plt.title(f'Tasa de Ahorro ($k\' / Rk$) para R = {rate} (Escala Lineal)', fontsize=16)
        plt.xlabel("Riqueza actual (k)", fontsize=12)
        plt.ylabel("Tasa de Ahorro", fontsize=12)
        
        # --- KEY CHANGES HERE ---
        # We remove the log scale and zoom in on a specific range
        plt.xlim(0, 2000) 
        
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, f"PHASE1_savings_rate_linear_R_{rate}.png"))
        plt.close()
        print(f"  > Saved LINEAR-SCALE savings rate plot for R={rate}")

### --- PHASE 2 & 3: COMPARATIVE ANALYSIS --- ###
def plot_time_series_comparison(data, metric, rate):
    """
    Plots the mean and 95% confidence interval for a given metric over time,
    comparing all experimental conditions.
    """
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=data[data['Rate'] == rate], x='Step', y=metric, hue='Experiment', palette='viridis')
    plt.title(f'{metric.replace("_", " ")} Over Time (R = {rate})', fontsize=16)
    plt.xlabel('Simulation Step', fontsize=12)
    plt.ylabel(metric.replace("_", " "), fontsize=12)
    plt.legend(title='Experiment')
    plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, f"PHASE2_3_TimeSeries_{metric}_R_{rate}.png"))
    plt.close()
    print(f"  > Saved time-series plot for {metric} at R={rate}")

def plot_final_distribution(data, metric, rate):
    """
    Creates a box plot comparing the distribution of the final values for a
    metric across all experimental conditions.
    """
    final_step_data = data[(data['Rate'] == rate) & (data['Step'] == SIMULATION_STEPS)]
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=final_step_data, x='Experiment', y=metric, palette='viridis')
    plt.title(f'Distribution of Final {metric.replace("_", " ")} (at Step {SIMULATION_STEPS}, R = {rate})', fontsize=16)
    plt.xlabel('Experiment', fontsize=12)
    plt.ylabel(f'Final {metric.replace("_", " ")}', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, f"PHASE2_3_FinalDist_{metric}_R_{rate}.png"))
    plt.close()
    print(f"  > Saved final distribution plot for {metric} at R={rate}")

def perform_statistical_analysis(data, metric, rate):
    """
    Performs ANOVA and post-hoc t-tests to check for significant differences
    between the baseline and other experiments.
    """
    print(f"\n--- Statistical Analysis for '{metric}' at R={rate} ---")
    final_step_data = data[(data['Rate'] == rate) & (data['Step'] == SIMULATION_STEPS)]
    
    experiments = final_step_data['Experiment'].unique()
    if len(experiments) < 2:
        print(f"  > Found only {len(experiments)} experiment group: {experiments}. ")
        print("  > Skipping statistical tests, as at least two groups are needed for comparison.")
        return # Exit the function early

    print(f"  > Found {len(experiments)} groups to compare: {list(experiments)}")
    baseline_data = final_step_data[final_step_data['Experiment'] == 'baseline'][metric]
    
    grouped_data = [final_step_data[final_step_data['Experiment'] == exp][metric] for exp in experiments]

    # 1. ANOVA - Checks if there is ANY significant difference among ANY of the groups.
    f_val, p_val_anova = stats.f_oneway(*grouped_data)
    print(f"One-Way ANOVA result: F-statistic = {f_val:.4f}, p-value = {p_val_anova:.4f}")
    
    if p_val_anova < 0.05:
        print("ANOVA is significant. Performing post-hoc t-tests against baseline...")
        # 2. T-Tests - Compare each social scenario to the baseline.
        for exp in experiments:
            if exp != 'baseline':
                exp_data = final_step_data[final_step_data['Experiment'] == exp][metric]
                t_stat, p_val_ttest = stats.ttest_ind(exp_data, baseline_data, equal_var=False) # Welch's t-test
                
                significance = "SIGNIFICANT" if p_val_ttest < 0.05 else "not significant"
                print(f"  - T-test '{exp}' vs 'baseline': p-value = {p_val_ttest:.4f} ({significance})")
    else:
        print("ANOVA is not significant. No strong evidence of differences between experiment groups.")

def plot_mechanism_dynamics(agent_data, rate):
    """
    Plots the evolution of agent-level variables that show social mechanisms at work.
    """
    print(f"\n--- Plotting Mechanism Dynamics for R={rate} ---")
    data_subset = agent_data[agent_data['Rate'] == rate]
    
    # Plot Beta Convergence (for Social Norms)
    if 'social_norms_only' in data_subset['Experiment'].unique():
        plt.figure(figsize=(14, 8))
        sns.lineplot(data=data_subset[data_subset['Experiment'].isin(['baseline', 'social_norms_only'])],
                     x='Step', y='Beta', hue='Original_Profile', style='Experiment')
        plt.title(f'Beta Convergence Under Social Norms (R = {rate})', fontsize=16)
        plt.ylabel('Beta (Present Bias Parameter)')
        plt.xlabel('Simulation Step')
        plt.ylim(0.5, 1.0) # Zoom in on the relevant range for beta
        plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, f"PHASE3_Mechanism_Beta_R_{rate}.png"))
        plt.close()
        print(f"  > Saved beta convergence plot for R={rate}")

    # Plot Financial Literacy Evolution (for Information Diffusion)
    if 'info_diffusion_only' in data_subset['Experiment'].unique():
        plt.figure(figsize=(14, 8))
        sns.lineplot(data=data_subset[data_subset['Experiment'].isin(['baseline', 'info_diffusion_only'])],
                     x='Step', y='Financial_Literacy', hue='Original_Profile', style='Experiment')
        plt.title(f'Financial Literacy Evolution (R = {rate})', fontsize=16)
        plt.ylabel('Financial Literacy Score')
        plt.xlabel('Simulation Step')
        plt.ylim(0, 1.05)
        plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, f"PHASE3_Mechanism_Literacy_R_{rate}.png"))
        plt.close()
        print(f"  > Saved financial literacy plot for R={rate}")
        
# =============================================================================
# --- 4. MAIN EXECUTION BLOCK ---
# =============================================================================
if __name__ == "__main__":
    # --- Load Data ---
    model_data, agent_data = load_and_aggregate_data(OUTPUT_DIR_CSV)
    print("\n--- Exporting fully aggregated data to CSV files... ---")
    agg_model_path = os.path.join(OUTPUT_DIR_PLOTS, "aggregated_model_data_all_runs.csv")
    agg_agent_path = os.path.join(OUTPUT_DIR_PLOTS, "aggregated_agent_data_all_runs.csv")
    
    model_data.to_csv(agg_model_path, index=False)
    agent_data.to_csv(agg_agent_path, index=False)
    
    print(f"  > Saved aggregated model data to: {agg_model_path}")
    print(f"  > Saved aggregated agent data to: {agg_agent_path}")

    # --- Run Phase 1 Analysis ---
    if os.path.exists(V_G_DIR):
        plot_phase1_validation(V_G_DIR, NUM_WEALTH_POINTS)
    else:
        print("Warning: v_g_functions directory not found. Skipping Phase 1 plots.")

    # --- Run Phase 2 & 3 Analysis for each interest rate ---
    for interest_rate in model_data['Rate'].unique():
        print(f"\n{'='*25} ANALYZING RESULTS FOR R = {interest_rate} {'='*25}")

        # Define key metrics to analyze
        model_metrics_to_plot = ["Average Wealth", "Gini_Coefficient", "Average Consumption", "Average Savings"]

        for metric in model_metrics_to_plot:
            # Generate comparative time-series plots
            plot_time_series_comparison(model_data, metric, interest_rate)
            
            # Generate final distribution box plots
            plot_final_distribution(model_data, metric, interest_rate)
            
            # Perform and print statistical tests
            perform_statistical_analysis(model_data, metric, interest_rate)

        # Plot the underlying social mechanisms from agent data
        plot_mechanism_dynamics(agent_data, interest_rate)

    print("\n--- Analysis complete. All plots saved to 'output_plots' directory. ---")