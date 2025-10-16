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
OUTPUT_DIR_TABLES = "output_tables"
V_G_DIR = os.path.join(OUTPUT_DIR_CSV, "v_g_functions")
os.makedirs(OUTPUT_DIR_PLOTS, exist_ok=True)
os.makedirs(OUTPUT_DIR_TABLES, exist_ok=True)

# --- Constants from experimental design ---
NUM_REPLICATIONS = 30
SIMULATION_STEPS = 200
NUM_WEALTH_POINTS = 1000

# --- Plotting Style ---
sns.set_theme(style="whitegrid")
palette = sns.color_palette("viridis", 5)

# --- Analysis Configuration ---
GENERATE_PHASE_1_PLOTS = False
EXPERIMENTS_TO_PLOT = []

# =============================================================================
# --- 2. DATA LOADING AND AGGREGATION ---
# =============================================================================
def load_and_aggregate_data(csv_dir):
    all_model_files = [f for f in os.listdir(csv_dir) if "model_data" in f]
    all_agent_files = [f for f in os.listdir(csv_dir) if "agent_data" in f]
    model_df_list = []
    agent_df_list = []
    pattern = re.compile(r"run_([a-zA-Z_]+)_R_(\d+\.\d+)_rep_(\d+)")

    print("--- Loading and parsing model data files... ---")
    for f in all_model_files:
        match = pattern.search(f)
        if match:
            run_name, rate, rep = match.groups()
            df = pd.read_csv(os.path.join(csv_dir, f))
            df = df.rename(columns={'Unnamed: 0': 'Step'})
            df['Experiment'] = run_name.rstrip('_')
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
        raise FileNotFoundError("No valid data files found.")

    aggregated_model_data = pd.concat(model_df_list, ignore_index=True)
    aggregated_agent_data = pd.concat(agent_df_list, ignore_index=True)
    print(f"Successfully loaded {len(aggregated_model_data)} model data rows and {len(aggregated_agent_data)} agent data rows.")
    return aggregated_model_data, aggregated_agent_data

# =============================================================================
# --- 3. TABLE GENERATION FUNCTIONS ---
# =============================================================================
def generate_summary_table(model_data, agent_data, rate):
    """
    Generates a summary table with initial and final values for key metrics.
    """
    print(f"--- Generating Summary Table for R = {rate} ---")
    model_rate_data = model_data[model_data['Rate'] == rate]
    agent_rate_data = agent_data[agent_data['Rate'] == rate]

    initial_model_data = model_rate_data[model_rate_data['Step'] == 0].groupby('Replication').first().reset_index()
    initial_agent_data = agent_rate_data[agent_rate_data['Step'] == 0]

    initial_avg_wealth = initial_model_data['Average Wealth'].mean()
    initial_avg_consumption = initial_model_data['Average Consumption'].mean()
    initial_gini = initial_model_data['Gini_Coefficient'].mean()
    initial_avg_beta = initial_agent_data['Beta'].mean()
    initial_avg_fin_lit = initial_agent_data['Financial_Literacy'].mean()

    step1_model_data = model_rate_data[model_rate_data['Step'] == 1].groupby('Replication').first().reset_index()
    initial_meaningful_savings = step1_model_data['Average Savings'].mean()

    final_step = model_rate_data['Step'].max()
    final_model_data = model_rate_data[model_rate_data['Step'] == final_step]
    final_agent_data = agent_rate_data[agent_rate_data['Step'] == final_step]

    table_rows = []
    experiments = sorted(final_model_data['Experiment'].unique())

    for exp in experiments:
        exp_final_model = final_model_data[final_model_data['Experiment'] == exp]
        exp_final_agent = final_agent_data[final_agent_data['Experiment'] == exp]

        final_avg_wealth = exp_final_model['Average Wealth'].mean()
        final_avg_consumption = exp_final_model['Average Consumption'].mean()
        final_avg_savings = exp_final_model['Average Savings'].mean()
        final_gini = exp_final_model['Gini_Coefficient'].mean()
        q90 = exp_final_model['Wealth_Quantile_90'].mean()
        q10 = exp_final_model['Wealth_Quantile_10'].mean()
        final_90_10_ratio = q90 / q10 if q10 > 0 else np.inf
        final_avg_beta = exp_final_agent['Beta'].mean()
        final_avg_fin_lit = exp_final_agent['Financial_Literacy'].mean()

        pct_delta_wealth = ((final_avg_wealth - initial_avg_wealth) / initial_avg_wealth) * 100
        pct_delta_consumption = ((final_avg_consumption - initial_avg_consumption) / initial_avg_consumption) * 100
        pct_delta_savings = ((final_avg_savings - initial_meaningful_savings) / initial_meaningful_savings) * 100 if initial_meaningful_savings != 0 else 0
        delta_gini = final_gini - initial_gini

        table_rows.append({
            'Experiment': exp,
            'Initial Avg. Wealth': initial_avg_wealth,
            'Avg. Wealth (Final)': final_avg_wealth,
            '% Δ Avg. Wealth (final-initial)': pct_delta_wealth,
            'Initial Avg. Consumption': initial_avg_consumption,
            'Avg. Consumption (Final)': final_avg_consumption,
            '% Δ Avg. Consumption (final-initial)': pct_delta_consumption,
            'Initial Avg. Savings': initial_meaningful_savings,
            'Avg. Savings (Final)': final_avg_savings,
            '% Δ Avg. Savings (initial vs final)': pct_delta_savings,
            'Initial Gini': initial_gini,
            'Gini (Final)': final_gini,
            'Δ Gini (Final-Initial)': delta_gini,
            'Final 90/10 Wealth Ratio': final_90_10_ratio,
            'Initial Avg. Beta': initial_avg_beta,
            'Final Avg. Beta': final_avg_beta,
            'Initial Fin. Literacy': initial_avg_fin_lit,
            'Final Avg. Fin. Literacy': final_avg_fin_lit
        })

    summary_df = pd.DataFrame(table_rows)
    filename = os.path.join(OUTPUT_DIR_TABLES, f"Table_Summary_Outcomes_R_{rate}.csv")
    summary_df.to_csv(filename, index=False, float_format='%.4f')
    print(f"  > Saved summary table to: {filename}")

def generate_statistical_table(model_data, agent_data, rate):
    """
    Generates the standardized statistical table with an F-statistic column,
    N/A for irrelevant Beta tests, and scientific notation for p-values.
    """
    print(f"--- Generating Standardized Statistical Table for R = {rate} ---")
    model_rate_data = model_data[model_data['Rate'] == rate]
    agent_rate_data = agent_data[agent_data['Rate'] == rate]
    
    final_step = model_rate_data['Step'].max()
    final_model_data = model_rate_data[model_rate_data['Step'] == final_step]
    final_agent_data = agent_rate_data[agent_rate_data['Step'] == final_step]
    
    if 'baseline' not in final_model_data['Experiment'].unique():
        print("  > Baseline data not found for this rate. Skipping statistical table.")
        return

    baseline_model_final = final_model_data[final_model_data['Experiment'] == 'baseline']
    baseline_agent_final = final_agent_data[final_agent_data['Experiment'] == 'baseline']
    
    stat_results = []
    
    experiments_to_compare = [exp for exp in sorted(final_model_data['Experiment'].unique()) if exp != 'baseline']
    all_experiments = sorted(final_model_data['Experiment'].unique())

    metrics_to_test = {
        'Final Avg. Wealth': ('model', 'Average Wealth'),
        'Final Gini Coeff.': ('model', 'Gini_Coefficient'),
        'Final Avg. Consumption': ('model', 'Average Consumption'),
        'Final Avg. Savings': ('model', 'Average Savings'),
        'Final Avg. Beta': ('agent', 'Beta')
    }

    for metric_name, (data_source, col_name) in metrics_to_test.items():
        grouped_data_for_anova = []
        if data_source == 'model':
            for exp in all_experiments:
                grouped_data_for_anova.append(final_model_data[final_model_data['Experiment'] == exp][col_name])
        else:
            for exp in all_experiments:
                grouped_data_for_anova.append(final_agent_data[final_agent_data['Experiment'] == exp].groupby('Replication')[col_name].mean())
        
        if len(grouped_data_for_anova) > 1:
            f_stat, _ = stats.f_oneway(*grouped_data_for_anova)
        else:
            f_stat = 'N/A'

        for exp in experiments_to_compare:
            if metric_name == 'Final Avg. Beta' and exp not in ['social_norms_only', 'all_interactions']:
                stat_results.append({
                    'Metric Tested': metric_name,
                    'Experiment Comparison': f"{exp} vs. Baseline",
                    'F-statistic': f_stat,
                    't-statistic': 'N/A',
                    'p-value': 'N/A',
                    'Significant (p < 0.05)?': 'N/A'
                })
                continue

            if data_source == 'model':
                group1 = final_model_data[final_model_data['Experiment'] == exp][col_name]
                group2 = baseline_model_final[col_name]
            else:
                group1 = final_agent_data[final_agent_data['Experiment'] == exp].groupby('Replication')[col_name].mean()
                group2 = baseline_agent_final.groupby('Replication')[col_name].mean()

            t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False, nan_policy='omit')
            stat_results.append({
                'Metric Tested': metric_name,
                'Experiment Comparison': f"{exp} vs. Baseline",
                'F-statistic': f_stat,
                't-statistic': t_stat,
                'p-value': p_val,
                'Significant (p < 0.05)?': 'Yes' if p_val < 0.05 else 'No'
            })

    stats_df = pd.DataFrame(stat_results)
    if not stats_df.empty:
        stats_df = stats_df[['Metric Tested', 'Experiment Comparison', 'F-statistic', 't-statistic', 'p-value', 'Significant (p < 0.05)?']]
    
    filename = os.path.join(OUTPUT_DIR_TABLES, f"Table_Statistical_Significance_R_{rate}.csv")
    stats_df.to_csv(filename, index=False, float_format='%.4e')
    print(f"  > Saved standardized statistical table with F-statistic to: {filename}")


# =============================================================================
# --- 4. ANALYSIS & PLOTTING FUNCTIONS ---
# =============================================================================
def plot_phase1_validation(v_g_dir, num_wealth_points):
    print("\n--- Generating Phase 1: VFI Validation Plots ---")
    agent_profiles = {
        "planner": {"beta": 0.97, "delta": 0.96}, "moderate": {"beta": 0.90, "delta": 0.91},
        "procrastinator": {"beta": 0.78, "delta": 0.95}, "inverse procrastinator": {"beta": 0.96, "delta": 0.85},
        "impulsive": {"beta": 0.60, "delta": 0.80}
    }
    interest_rates = [1.05, 1.12]
    wealth_grid = np.geomspace(1e-6, 1000001, num_wealth_points)

    for rate in interest_rates:
        plt.figure(figsize=(14, 8))
        for name, params in agent_profiles.items():
            filepath = os.path.join(v_g_dir, f"R_{rate}_{name}_policy_function.npy")
            if os.path.exists(filepath):
                g_func = np.load(filepath)
                R_star = 1 + (1 - params['delta']) / (params['beta'] * params['delta'])
                plt.plot(wealth_grid, g_func, label=f'{name.title()} (R* ≈ {R_star:.2f})')
        plt.plot(wealth_grid, wealth_grid, 'k--', label="k' = k (Zero Net Savings)", alpha=0.6)
        plt.title(f'Saving Policy Functions g(k) for R = {rate} (log-log scale)', fontsize=16)
        plt.xlabel("Current Wealth (k)", fontsize=12)
        plt.ylabel("Next Period Wealth (k')", fontsize=12)
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, f"PHASE1_policy_functions_log_R_{rate}.png"))
        plt.close()
        print(f"  > Saved LOG-SCALE policy function plot for R={rate}")

def plot_time_series_comparison(data, metric, rate):
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
    final_step_data = data[(data['Rate'] == rate) & (data['Step'] == SIMULATION_STEPS - 1)]
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=final_step_data, x='Experiment', y=metric, palette='viridis')
    plt.title(f'Final Distribution of {metric.replace("_", " ")} (Step {SIMULATION_STEPS}, R = {rate})', fontsize=16)
    plt.xlabel('Experiment', fontsize=12)
    plt.ylabel(f'Final {metric.replace("_", " ")}', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, f"PHASE2_3_FinalDist_{metric}_R_{rate}.png"))
    plt.close()
    print(f"  > Saved final distribution plot for {metric} at R={rate}")

# --- NEWLY ADDED PLOTTING FUNCTIONS ---
def plot_wealth_by_profile(agent_data, rate):
    """
    Plots the average wealth trajectory over time, segmented by agent profile.
    """
    plt.figure(figsize=(14, 8))
    data_subset = agent_data[agent_data['Rate'] == rate]
    sns.lineplot(data=data_subset, x='Step', y='Wealth', hue='Original_Profile', style='Experiment')
    plt.title(f'Average Wealth Trajectory by Agent Profile (R = {rate})', fontsize=16)
    plt.xlabel('Simulation Step', fontsize=12)
    plt.ylabel('Average Wealth', fontsize=12)
    plt.legend(title='Agent Profile & Experiment')
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, f"PHASE2_3_WealthByProfile_R_{rate}.png"))
    plt.close()
    print(f"  > Saved agent wealth by profile plot for R={rate}")

def plot_final_wealth_distribution_histogram(agent_data, rate):
    """
    Creates a histogram on a LOG scale showing the initial vs. final distribution of wealth.
    """
    final_step_agent_data = agent_data[(agent_data['Rate'] == rate) & (agent_data['Step'] == SIMULATION_STEPS - 1)]
    initial_step_agent_data = agent_data[(agent_data['Rate'] == rate) & (agent_data['Step'] == 0)]
    plt.figure(figsize=(14, 8))
    sns.histplot(data=initial_step_agent_data, x='Wealth', color="grey", alpha=0.5, 
                 log_scale=True, label='Initial Distribution (t=0)')
    experiments = final_step_agent_data['Experiment'].unique()
    colors = sns.color_palette('viridis', n_colors=len(experiments))
    for i, exp_name in enumerate(experiments):
        exp_data = final_step_agent_data[final_step_agent_data['Experiment'] == exp_name]
        label = f'Final (t={SIMULATION_STEPS}): {exp_name}'
        sns.histplot(data=exp_data, x='Wealth', color=colors[i], 
                     alpha=0.5, log_scale=True, label=label, 
                     element="step", kde=True)
    plt.title(f'Initial vs. Final Wealth Distribution (Log Scale), R = {rate})', fontsize=16)
    plt.xlabel('Wealth', fontsize=12)
    plt.ylabel('Number of Agents', fontsize=12)
    plt.legend() 
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, f"PHASE2_3_FinalWealthHistogram_WithInitial_R_{rate}.png"))
    plt.close()
    print(f"  > Saved final agent wealth histogram (log scale) for R={rate}")

def plot_final_wealth_distribution_linear(agent_data, rate):
    """
    Creates a histogram on a LINEAR scale showing the initial vs. final distribution of wealth.
    """
    final_step_agent_data = agent_data[(agent_data['Rate'] == rate) & (agent_data['Step'] == SIMULATION_STEPS - 1)]
    initial_step_agent_data = agent_data[(agent_data['Rate'] == rate) & (agent_data['Step'] == 0)]
    plt.figure(figsize=(14, 8))
    sns.histplot(data=initial_step_agent_data, x='Wealth', color="grey", alpha=0.5, 
                 label='Initial Distribution (t=0)')
    experiments = final_step_agent_data['Experiment'].unique()
    colors = sns.color_palette('viridis', n_colors=len(experiments))
    for i, exp_name in enumerate(experiments):
        exp_data = final_step_agent_data[final_step_agent_data['Experiment'] == exp_name]
        label = f'Final (t={SIMULATION_STEPS}): {exp_name}'
        sns.histplot(data=exp_data, x='Wealth', color=colors[i], 
                     alpha=0.5, label=label, element="step")
    plt.title(f'Initial vs. Final Wealth Distribution (Linear Scale), R = {rate})', fontsize=16)
    plt.xlabel('Wealth', fontsize=12)
    plt.ylabel('Number of Agents', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, f"PHASE2_3_FinalWealthHistogram_LINEAR_WithInitial_R_{rate}.png"))
    plt.close()
    print(f"  > Saved final agent wealth histogram (linear scale) for R={rate}")
# ------------------------------------

def perform_statistical_analysis(data, metric, rate):
    print(f"\n--- Statistical Analysis for '{metric}' at R={rate} (Console Output) ---")
    final_step_data = data[(data['Rate'] == rate) & (data['Step'] == SIMULATION_STEPS - 1)]
    experiments = final_step_data['Experiment'].unique()
    if len(experiments) < 2:
        print(f"  > Only one experiment group found. Skipping tests.")
        return
    if 'baseline' not in experiments:
        print("  > Baseline experiment not found. Skipping t-tests.")
        return
    baseline_data = final_step_data[final_step_data['Experiment'] == 'baseline'][metric]
    grouped_data = [final_step_data[final_step_data['Experiment'] == exp][metric] for exp in experiments]
    f_val, p_val_anova = stats.f_oneway(*grouped_data)
    print(f"One-Way ANOVA result: F-statistic = {f_val:.4f}, p-value = {p_val_anova:.4f}")
    if p_val_anova < 0.05:
        print("ANOVA is significant. Performing t-tests against baseline...")
        for exp in experiments:
            if exp != 'baseline':
                exp_data = final_step_data[final_step_data['Experiment'] == exp][metric]
                t_stat, p_val_ttest = stats.ttest_ind(exp_data, baseline_data, equal_var=False)
                significance = "SIGNIFICANT" if p_val_ttest < 0.05 else "not significant"
                print(f"  - T-test '{exp}' vs 'baseline': p-value = {p_val_ttest:.4f} ({significance})")
    else:
        print("ANOVA is not significant.")

# =============================================================================
# --- 5. MAIN EXECUTION BLOCK ---
# =============================================================================
if __name__ == "__main__":
    agg_model_path = os.path.join(OUTPUT_DIR_PLOTS, "aggregated_model_data_all_runs.csv")
    agg_agent_path = os.path.join(OUTPUT_DIR_PLOTS, "aggregated_agent_data_all_runs.csv")

    if os.path.exists(agg_model_path) and os.path.exists(agg_agent_path):
        print("--- Found existing aggregated data files. Loading directly. ---")
        model_data = pd.read_csv(agg_model_path)
        agent_data = pd.read_csv(agg_agent_path)
    else:
        print("--- No aggregated data found. Running full aggregation process. ---")
        model_data, agent_data = load_and_aggregate_data(OUTPUT_DIR_CSV)
        model_data.to_csv(agg_model_path, index=False)
        agent_data.to_csv(agg_agent_path, index=False)

    if EXPERIMENTS_TO_PLOT:
        print(f"\n--- Filtering data to include only: {EXPERIMENTS_TO_PLOT} ---")
        model_data = model_data[model_data['Experiment'].isin(EXPERIMENTS_TO_PLOT)].copy()
        agent_data = agent_data[agent_data['Experiment'].isin(EXPERIMENTS_TO_PLOT)].copy()
        if model_data.empty:
            raise ValueError("Filtering resulted in empty dataframes.")
    else:
        print("\n--- No specific experiments selected. Analyzing all available experiments. ---")

    if GENERATE_PHASE_1_PLOTS:
        if os.path.exists(V_G_DIR):
            plot_phase1_validation(V_G_DIR, NUM_WEALTH_POINTS)
        else:
            print("Warning: v_g_functions directory not found. Skipping Phase 1 plots.")
    else:
        print("\n--- Skipping Phase 1 VFI Validation Plots as per configuration. ---")

    for interest_rate in sorted(model_data['Rate'].unique()):
        print(f"\n{'='*25} ANALYZING RESULTS FOR R = {interest_rate} {'='*25}")
        generate_summary_table(model_data, agent_data, interest_rate)
        generate_statistical_table(model_data, agent_data, interest_rate)
        
        print(f"\n--- Generating Plots for R = {interest_rate} ---")
        model_metrics_to_plot = ["Average Wealth", "Gini_Coefficient", "Average Consumption", "Average Savings"]
        for metric in model_metrics_to_plot:
            plot_time_series_comparison(model_data, metric, interest_rate)
            plot_final_distribution(model_data, metric, interest_rate)
            perform_statistical_analysis(model_data, metric, interest_rate)

        # --- CALLS TO THE RE-ADDED PLOTTING FUNCTIONS ---
        plot_wealth_by_profile(agent_data, interest_rate)
        plot_final_wealth_distribution_histogram(agent_data, interest_rate)
        plot_final_wealth_distribution_linear(agent_data, interest_rate)

    print("\n--- Analysis complete. All plots and tables saved. ---")