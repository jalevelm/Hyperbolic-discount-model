# analyze_results.py - Data processing, statistical testing, and visualization pipeline
# Copyright (C) 2026 Alejandro Velazquez
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import re
from scipy import stats
from matplotlib.patches import Patch 
import pingouin as pg

# --- 1. SETUP & CONFIGURATION ---

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
GENERATE_PLOTS = True
EXPERIMENTS_TO_PLOT = ["baseline", "info_diffusion_only", "peer_comparison_only", "social_norms_only", "all_interactions"] 
RATES_TO_PROCESS = [1.12] 


# --- 2. DATA LOADING AND AGGREGATION ---

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
            df = df.rename(columns={'Unnamed: 0': 'Step', 'Unnamed: 1': 'AgentID'})
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


# --- 3. TABLE GENERATION FUNCTIONS ---

def generate_summary_table(model_data, agent_data, rate):
    """
    Generates a summary table with initial/final values, broke agents, and wealth distribution stats.
    """
    print(f"--- Generating Summary Table for R = {rate} ---")
    model_rate_data = model_data[model_data['Rate'] == rate]
    agent_rate_data = agent_data[agent_data['Rate'] == rate]

    initial_model_data = model_rate_data[model_rate_data['Step'] == 0].groupby('Replication').first().reset_index()
    initial_agent_data = agent_rate_data[agent_rate_data['Step'] == 0]

    initial_avg_wealth = initial_model_data['Average Wealth'].mean()
    initial_median_wealth = initial_agent_data['Wealth'].median()
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

        # --- Standard Metrics ---
        final_avg_wealth = exp_final_model['Average Wealth'].mean()
        final_median_wealth = exp_final_agent['Wealth'].median()
        final_std_dev_wealth = exp_final_agent['Wealth'].std()
        final_avg_consumption = exp_final_model['Average Consumption'].mean()
        final_avg_savings = exp_final_model['Average Savings'].mean()
        final_gini = exp_final_model['Gini_Coefficient'].mean()
        q90 = exp_final_model['Wealth_Quantile_90'].mean()
        q10 = exp_final_model['Wealth_Quantile_10'].mean()
        final_90_10_ratio = q90 / q10 if q10 > 0 else np.inf
        final_avg_beta = exp_final_agent['Beta'].mean()
        final_avg_fin_lit = exp_final_agent['Financial_Literacy'].mean()

        # --- Deltas ---
        pct_delta_wealth = ((final_avg_wealth - initial_avg_wealth) / initial_avg_wealth) * 100
        pct_delta_consumption = ((final_avg_consumption - initial_avg_consumption) / initial_avg_consumption) * 100
        pct_delta_savings = ((final_avg_savings - initial_meaningful_savings) / initial_meaningful_savings) * 100 if initial_meaningful_savings != 0 else 0
        delta_gini = final_gini - initial_gini
        
        # --- Broke Agents ---
        total_final_agents = len(exp_final_agent)
        broke_agents_count = len(exp_final_agent[exp_final_agent['Wealth'] < 0.1])
        percentage_broke = (broke_agents_count / total_final_agents) * 100 if total_final_agents > 0 else 0

        table_rows.append({
            'Experiment': exp,
            'Initial Avg. Wealth': initial_avg_wealth,
            'Avg. Wealth (Final)': final_avg_wealth,
            '% Δ Avg. Wealth (final-initial)': pct_delta_wealth,
            'Initial Median Wealth': initial_median_wealth,
            'Median Wealth (Final)': final_median_wealth,
            'Std. Dev. Wealth (Final)': final_std_dev_wealth,
            'Initial Avg. Consumption': initial_avg_consumption,
            'Avg. Consumption (Final)': final_avg_consumption,
            '% Δ Avg. Consumption (final-initial)': pct_delta_consumption,
            'Initial Avg. Savings': initial_meaningful_savings,
            'Avg. Savings (Final)': final_avg_savings,
            '% Δ Avg. Savings (initial vs final)': pct_delta_savings,
            'Initial Gini': initial_gini,
            'Gini (Final)': final_gini,
            'Δ Gini (Final-Initial)': delta_gini,
            '% Broke Agents (Final)': percentage_broke,
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
    Generates the standardized statistical table using Welch's ANOVA and Games-Howell post-hoc.
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
        f_stat = 'N/A'
        p_anova = 'N/A'
        beta_modifying_experiments = {'social_norms_only', 'all_interactions'}
        
        is_beta_metric_and_relevant = (metric_name == 'Final Avg. Beta' and any(exp in beta_modifying_experiments for exp in all_experiments))
        is_not_beta_metric = metric_name != 'Final Avg. Beta'

        if data_source == 'model':
            df_metric = final_model_data[['Experiment', col_name]].copy()
            df_metric = df_metric[df_metric['Experiment'].isin(all_experiments)]
        else:
            df_ag = final_agent_data[final_agent_data['Experiment'].isin(all_experiments)]
            df_metric = df_ag.groupby(['Experiment', 'Replication'])[col_name].mean().reset_index()

        posthoc = None

        if is_not_beta_metric or is_beta_metric_and_relevant:
            if len(all_experiments) > 1:

                df_metric[col_name] = df_metric[col_name].astype(float) + np.random.uniform(-1e-4, 1e-4, size=len(df_metric))
                
                try:

                    anova_res = pg.welch_anova(dv=col_name, between='Experiment', data=df_metric)
                    f_stat = anova_res['F'].iloc[0]
                    p_anova = anova_res['p_unc'].iloc[0]
                    
                    if p_anova < 0.05:
                        posthoc = pg.pairwise_gameshowell(dv=col_name, between='Experiment', data=df_metric)
                except Exception as e:
                    print(f"  > Advertencia: No se pudo calcular ANOVA/Games-Howell para {metric_name}. Motivo: {e}")

        for exp in experiments_to_compare:
  
            if metric_name == 'Final Avg. Beta' and exp not in beta_modifying_experiments:
                stat_results.append({
                    'Metric Tested': metric_name,
                    'Experiment Comparison': f"{exp} vs. Baseline",
                    'F-statistic (Welch)': f_stat,
                    't-statistic (Games-Howell)': 'N/A',
                    'p-value': 'N/A',
                    'Significant (p < 0.05)?': 'N/A'
                })
                continue

            t_stat_gh = 'N/A'
            p_val_gh = 'N/A'
            is_sig = 'No'

            if posthoc is not None and not posthoc.empty:

                match = posthoc[((posthoc['A'] == exp) & (posthoc['B'] == 'baseline')) | 
                                ((posthoc['A'] == 'baseline') & (posthoc['B'] == exp))]
                
                if not match.empty:

                    t_stat_gh = match['T'].iloc[0]
                    if match['A'].iloc[0] == 'baseline':
                        t_stat_gh = -t_stat_gh 

                    p_val_gh = match['pval'].iloc[0]
                    is_sig = 'Yes' if p_val_gh < 0.05 else 'No'
            elif p_anova != 'N/A' and p_anova >= 0.05:
                p_val_gh = '> 0.05 (ns ANOVA)'

            stat_results.append({
                'Metric Tested': metric_name,
                'Experiment Comparison': f"{exp} vs. Baseline",
                'F-statistic (Welch)': f_stat,
                't-statistic (Games-Howell)': t_stat_gh,
                'p-value': p_val_gh,
                'Significant (p < 0.05)?': is_sig
            })

    stats_df = pd.DataFrame(stat_results)
    if not stats_df.empty:
        stats_df = stats_df[['Metric Tested', 'Experiment Comparison', 'F-statistic (Welch)', 't-statistic (Games-Howell)', 'p-value', 'Significant (p < 0.05)?']]
    
    filename = os.path.join(OUTPUT_DIR_TABLES, f"Table_Statistical_Significance_R_{rate}.csv")
    stats_df.to_csv(filename, index=False, float_format='%.4e')
    print(f"  > Saved standardized statistical table with Welch F-statistic to: {filename}")


# --- 4. ANALYSIS & PLOTTING FUNCTIONS ---

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
        plt.plot(wealth_grid, wealth_grid, 'k--', label="k' = k (Ahorro Neto Cero)", alpha=0.6)
        plt.title(f'Funciones de Política de Ahorro g(k) para R = {rate} (escala log-log)', fontsize=16)
        plt.xlabel("Riqueza Actual (k)", fontsize=12)
        plt.ylabel("Riqueza del Siguiente Periodo (k')", fontsize=12)
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, f"PHASE1_policy_functions_log_R_{rate}.png"))
        plt.close()
        print(f"  > Saved LOG-SCALE policy function plot for R={rate}")

def plot_time_series_comparison(data, metric, rate, target_ax=None):
    if target_ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
        save_plot = True
    else:
        ax = target_ax
        save_plot = False

    plot_data = data[data['Rate'] == rate]
    num_experiments = plot_data['Experiment'].nunique()
    
    if num_experiments >= 5:
        palette_to_use = 'tab10'
        ci_to_use = None
    else:
        palette_to_use = 'viridis'
        ci_to_use = 'sd'
        
    sns.lineplot(data=plot_data, x='Step', y=metric, hue='Experiment', palette=palette_to_use, ci=ci_to_use, ax=ax)
    
    ax.set_xlabel('Paso de Simulación', fontsize=11)
    ax.set_ylabel(metric.replace('_', ' '), fontsize=11)

    ax.legend(title='Experimento')
    
    if save_plot:
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, f'PHASE2_3_TimeSeries_{metric}_R_{rate}.png'))
        plt.close()

def plot_final_distribution(data, metric, rate):
    final_step_data = data[(data['Rate'] == rate) & (data['Step'] == SIMULATION_STEPS - 1)]
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=final_step_data, x='Experiment', y=metric, palette='viridis')
    plt.title(f'Distribución Final de {metric.replace("_", " ")} (Paso {SIMULATION_STEPS}, R = {rate})', fontsize=16)
    plt.xlabel('Experimento', fontsize=12)
    plt.ylabel(f'{metric.replace("_", " ")} Final', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, f"PHASE2_3_FinalDist_{metric}_R_{rate}.png"))
    plt.close()
    print(f"  > Saved final distribution plot for {metric} at R={rate}")

def plot_wealth_by_profile(agent_data, rate):
    data_subset = agent_data[agent_data['Rate'] == rate]
    
    nombres_mecanismos = {
        "info_diffusion_only": "Difusión de Información",
        "peer_comparison_only": "Comparación entre Pares",
        "social_norms_only": "Normas Sociales",
        "all_interactions": "Todas las Interacciones"
    }
    
    experimentos_validos = {k: v for k, v in nombres_mecanismos.items() if k in data_subset['Experiment'].unique()}
    
    if 'baseline' not in data_subset['Experiment'].unique() or not experimentos_validos:
        print("  > Faltan datos o 'baseline' para generar la cuadrícula. Omitiendo.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    panel_labels = ['A', 'B', 'C', 'D']
    
    global_profile_handles = []
    global_profile_labels = []
    
    for i, (exp_key, exp_name) in enumerate(experimentos_validos.items()):
        ax = axes[i]
        
        exp_data = data_subset[data_subset['Experiment'].isin(['baseline', exp_key])]
        
        estilos_lineas = {exp_key: '', 'baseline': (4, 2)}
        
        sns.lineplot(data=exp_data, 
                     x='Step', 
                     y='Wealth', 
                     hue='Original_Profile',  
                     style='Experiment', 
                     dashes=estilos_lineas, 
                     palette='tab10', 
                     errorbar=None,
                     ax=ax) 
        
        ax.set_title("") 
        ax.set_xlabel('Paso de Simulación', fontsize=12)
        ax.set_ylabel('Riqueza Promedio', fontsize=12)
        
        ax.text(-0.06, 1.02, panel_labels[i], transform=ax.transAxes, 
                fontsize=16, fontweight='bold', va='bottom')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, which="both", ls="--")
        
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.get_legend().remove()
            
            line_handles = []
            line_labels = []
            
            is_experiment = False
            for h, l in zip(handles, labels):
                if l == 'Original_Profile':
                    is_experiment = False
                    continue
                elif l == 'Experiment':
                    is_experiment = True
                    continue
                    
                if is_experiment:
                    if l == exp_key:
                        line_handles.append(h)
                        line_labels.append(exp_name) 
                    elif l == 'baseline':
                        line_handles.append(h)
                        line_labels.append('Línea Base')
                else:
                    if i == 0: 
                        global_profile_handles.append(h)
                        global_profile_labels.append(str(l).title())
                        
            local_legend = ax.legend(line_handles, line_labels, title='LÍNEAS:', 
                                     loc='upper left', frameon=True, 
                                     fontsize=10, title_fontsize=11)
            local_legend.get_frame().set_edgecolor('black')
            local_legend.get_frame().set_linewidth(0.5)
            
    for j in range(len(experimentos_validos), len(axes)):
        axes[j].set_visible(False)
        
    if global_profile_handles:
        fig.legend(global_profile_handles, global_profile_labels, title='PERFILES:', 
                   loc='lower center', bbox_to_anchor=(0.5, 0.95), 
                   ncol=5, frameon=False, fontsize=11, title_fontsize=12)
        
    plt.subplots_adjust(top=0.90, hspace=0.3, wspace=0.2)
    
    filename = f"Panel_Trayectorias_Perfiles_2x2_R_{rate}.png"
    fig.savefig(os.path.join(OUTPUT_DIR_PLOTS, filename), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  > Guardado Panel 2x2 de Trayectorias: {filename}")

def plot_beta_by_profile(agent_data, rate):
    allowed_exps = ['baseline', 'social_norms_only', 'all_interactions']
    data_subset = agent_data[(agent_data['Rate'] == rate) & (agent_data['Experiment'].isin(allowed_exps))]
    
    if data_subset.empty:
        print(f"  > No hay datos para los experimentos requeridos en Beta para R={rate}.")
        return
        
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=data_subset, x='Step', y='Beta', hue='Original_Profile', style='Experiment', palette='tab10', ci=None)
    
    plt.xlabel('Paso de Simulación', fontsize=12)
    plt.ylabel('Beta Promedio', fontsize=12)
    plt.legend(title='Perfil y Experimento', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    
    filename = f"PHASE2_3_BetaByProfile_Filtrado_R_{rate}.png"
    plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, filename))
    plt.close()
    print(f"  > Guardada trayectoria de Beta filtrada para R={rate}")


def plot_final_wealth_distribution_barplot(agent_data, rate, log_scale=True, target_ax=None):
    final_step_agent_data = agent_data[(agent_data['Rate'] == rate) & (agent_data['Step'] == SIMULATION_STEPS - 1)].copy()
    experiments = sorted(final_step_agent_data['Experiment'].unique())
    if not experiments:
        return
    
    initial_step_agent_data = agent_data[(agent_data['Rate'] == rate) & (agent_data['Step'] == 0) & (agent_data['Experiment'] == experiments[0])].copy()

    unique_replications = final_step_agent_data['Replication'].unique()
    if 'baseline' in experiments:
        non_baseline = final_step_agent_data[final_step_agent_data['Experiment'] != 'baseline']
        if not non_baseline.empty:
            min_reps_exp = non_baseline.groupby('Experiment')['Replication'].nunique().idxmin()
            matching_reps = non_baseline[non_baseline['Experiment'] == min_reps_exp]['Replication'].unique()
            final_step_agent_data = final_step_agent_data[final_step_agent_data['Replication'].isin(matching_reps)]
            initial_step_agent_data = initial_step_agent_data[initial_step_agent_data['Replication'].isin(matching_reps)]
            unique_replications = matching_reps

    FLOOR_VALUE = 1e-2 
    final_step_agent_data['Wealth'] = np.clip(final_step_agent_data['Wealth'], a_min=FLOOR_VALUE, a_max=None)
    initial_step_agent_data['Wealth'] = np.clip(initial_step_agent_data['Wealth'], a_min=FLOOR_VALUE, a_max=None) 
    
    binned_data_all_reps = []
    binned_data_initial = []
    
    if log_scale:
        plot_filename_suffix = "LOG"
        bin_edges = np.power(10.0, np.arange(-2, 8))
        bin_labels = [f"$10^{{{i}}}$ a $10^{{{i+1}}}$" for i in range(-2, 7)]
        bin_order = bin_labels
        
        for (exp, rep), group_data in final_step_agent_data.groupby(['Experiment', 'Replication']):
            binned = pd.cut(group_data['Wealth'], bins=bin_edges, labels=bin_labels, right=False)
            counts = binned.value_counts().reset_index()
            counts.columns = ['Wealth Bin Label', 'Count']
            counts['Experiment'] = exp
            counts['Replication'] = rep
            binned_data_all_reps.append(counts)
            
        for rep, group_data in initial_step_agent_data.groupby('Replication'):
            binned = pd.cut(group_data['Wealth'], bins=bin_edges, labels=bin_labels, right=False)
            counts = binned.value_counts().reset_index()
            counts.columns = ['Wealth Bin Label', 'Count']
            counts['Replication'] = rep
            binned_data_initial.append(counts)
            
        if not binned_data_all_reps: return
        # CORRECCIÓN AQUÍ: Usamos concat para la lista de DataFrames
        binned_df = pd.concat(binned_data_all_reps, ignore_index=True)
        binned_df_initial = pd.concat(binned_data_initial, ignore_index=True) if binned_data_initial else pd.DataFrame()
            
    else:
        plot_filename_suffix = "LINEAR"
        data_min = FLOOR_VALUE
        data_max = final_step_agent_data['Wealth'].max()
        if pd.isna(data_max) or data_max <= data_min: data_max = data_min + 1
        bins = np.linspace(data_min, data_max, 25)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        for exp in experiments:
            for rep in unique_replications:
                rep_data = final_step_agent_data[(final_step_agent_data['Experiment'] == exp) & (final_step_agent_data['Replication'] == rep)]['Wealth']
                if not rep_data.empty:
                    counts, _ = np.histogram(rep_data, bins=bins)
                    for i, count in enumerate(counts):
                        binned_data_all_reps.append({'Experiment': exp, 'Replication': rep, 'Wealth Bin Label': f'{bin_centers[i]:.1e}', 'Count': count})
        
        for rep in unique_replications:
            rep_data = initial_step_agent_data[initial_step_agent_data['Replication'] == rep]['Wealth']
            if not rep_data.empty:
                counts, _ = np.histogram(rep_data, bins=bins)
                for i, count in enumerate(counts):
                    binned_data_initial.append({'Replication': rep, 'Wealth Bin Label': f'{bin_centers[i]:.1e}', 'Count': count})

        if not binned_data_all_reps: return
        # CORRECCIÓN AQUÍ: Usamos DataFrame para la lista de diccionarios
        binned_df = pd.DataFrame(binned_data_all_reps)
        binned_df_initial = pd.DataFrame(binned_data_initial)
        bin_order = binned_df['Wealth Bin Label'].unique()
        
    if target_ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
        save_plot = True
    else:
        ax = target_ax
        save_plot = False

    palette_to_use = "tab10" if len(experiments) >= 5 else "viridis"

    sns.barplot(data=binned_df, x='Wealth Bin Label', y='Count', hue='Experiment', order=bin_order, palette=palette_to_use, errorbar='sd', ax=ax)

    if not binned_df_initial.empty:
        initial_stats = binned_df_initial.groupby('Wealth Bin Label')['Count'].agg(['mean', 'std']).reindex(bin_order)
        xticks_locs = ax.get_xticks()
        if len(xticks_locs) == len(bin_order):
            label_to_loc = dict(zip(bin_order, xticks_locs))
            for bin_label in bin_order:
                if bin_label in initial_stats.index and not pd.isna(initial_stats.loc[bin_label, 'mean']):
                    m = initial_stats.loc[bin_label, 'mean']
                    s = initial_stats.loc[bin_label, 'std']
                    x_pos = label_to_loc[bin_label]
                    ax.bar(x_pos, m, width=0.8, facecolor=(1,1,1,0), edgecolor='black', linewidth=1.0, label='_nolegend_')
                    ax.errorbar(x_pos, m, yerr=s, fmt='none', ecolor='black', elinewidth=1.0, capsize=3, label='_nolegend_')

    ax.set_title("") 
    ax.set_xlabel('Riqueza (Escala Log)' if log_scale else 'Riqueza (Escala Lineal)', fontsize=12)
    ax.set_ylabel('Número de Agentes', fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    
    handles, labels = ax.get_legend_handles_labels()
    if not binned_df_initial.empty and 'Initial (t=0)' not in labels:
        handles.append(Patch(facecolor='none', edgecolor='black', linewidth=1.0, label='Initial (t=0)'))
        labels.append('Initial (t=0)')
    
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Experimento', bbox_to_anchor=(1.01, 1), loc='upper left')
    
    if save_plot:
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, f"PHASE2_3_FinalWealthDist_Barplot_{plot_filename_suffix}_R_{rate}.png"), bbox_inches='tight')
        plt.close()

def plot_final_consumption_distribution_barplot(agent_data, rate, target_ax=None):
    final_step_agent_data = agent_data[(agent_data['Rate'] == rate) & (agent_data['Step'] == SIMULATION_STEPS - 1)].copy()
    experiments = sorted(final_step_agent_data['Experiment'].unique())
    if not experiments: return
    
    if 'baseline' in experiments:
        non_baseline = final_step_agent_data[final_step_agent_data['Experiment'] != 'baseline']
        if not non_baseline.empty:
            min_reps_exp = non_baseline.groupby('Experiment')['Replication'].nunique().idxmin()
            matching_reps = non_baseline[non_baseline['Experiment'] == min_reps_exp]['Replication'].unique()
            final_step_agent_data = final_step_agent_data[final_step_agent_data['Replication'].isin(matching_reps)]

    final_step_agent_data['Consumption'] = np.clip(final_step_agent_data['Consumption'], a_min=1e-2, a_max=None)
    
    bin_edges = np.power(10.0, np.arange(-2, 8))
    bin_labels = [f"$10^{{{i}}}$ - $10^{{{i+1}}}$" for i in range(-2, 7)]
    
    binned_data = []
    for (exp, rep), group in final_step_agent_data.groupby(['Experiment', 'Replication']):
        counts = pd.cut(group['Consumption'], bins=bin_edges, labels=bin_labels, right=False).value_counts().reset_index()
        counts.columns = ['Consumption Bin Label', 'Count']
        counts['Experiment'] = exp
        binned_data.append(counts)
        
    binned_df = pd.concat(binned_data, ignore_index=True)
    
    if target_ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
        save_plot = True
    else:
        ax = target_ax
        save_plot = False

    palette_to_use = 'tab10' if len(experiments) >= 5 else 'viridis'
    sns.barplot(data=binned_df, x='Consumption Bin Label', y='Count', hue='Experiment', order=bin_labels, palette=palette_to_use, ci='sd', ax=ax)
    
    ax.set_title("") 
    ax.set_xlabel('Consumo (Escala Log)', fontsize=12)
    ax.set_ylabel('Número de Agentes', fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.legend(title='Experimento', bbox_to_anchor=(1.01, 1), loc='upper left')
    
    if save_plot:
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, f'PHASE2_3_FinalConsumptionDist_Barplot_LOG_R_{rate}.png'), bbox_inches='tight')
        plt.close()

def plot_composite_time_series(data, rate):
    """Crea panel 3x1 (vertical) para Riqueza, Consumo y Gini con formato APA (A, B, C)"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 20)) 
    
    metrics = ["Average Wealth", "Average Consumption", "Gini_Coefficient"]
    panel_labels = ['A', 'B', 'C']
    
    for i, metric in enumerate(metrics):
        plot_time_series_comparison(data, metric, rate, target_ax=axes[i])
        
        axes[i].text(-0.06, 1.02, panel_labels[i], transform=axes[i].transAxes, 
                     fontsize=16, fontweight='bold', va='bottom')
        
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        
        if i < 2:
            legend = axes[i].get_legend()
            if legend:
                legend.remove()
        else:
            axes[i].legend(title='Experimento', bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
                
    plt.subplots_adjust(hspace=0.3) 
    filename = f'Panel_Agregados_Vertical_Riqueza_Consumo_Gini_R_{rate}.png'
    
    fig.savefig(os.path.join(OUTPUT_DIR_PLOTS, filename), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'  > Guardado Panel Vertical de Agregados: {filename}')


def plot_composite_distributions(agent_data, rate):
    """Crea panel 2x1 para Distribuciones con formato APA (A, B)"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 16)) 
    
    # Riqueza LOG
    plot_final_wealth_distribution_barplot(agent_data, rate, log_scale=True, target_ax=axes[0])
    # Consumo LOG
    plot_final_consumption_distribution_barplot(agent_data, rate, target_ax=axes[1])
    
    panel_labels = ['A', 'B']
    
    for i, ax in enumerate(axes):
        # Añadir la letra identificadora del panel (Formato APA)
        ax.text(-0.06, 1.02, panel_labels[i], transform=ax.transAxes, 
                fontsize=16, fontweight='bold', va='bottom')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if ax.get_legend() is not None:
             ax.get_legend().set_frame_on(False)
    
    plt.subplots_adjust(hspace=0.4) 
    
    filename = f'Panel_Distribuciones_Vertical_Log_R_{rate}.png'
    fig.savefig(os.path.join(OUTPUT_DIR_PLOTS, filename), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'  > Guardado Panel Vertical de Distribuciones: {filename}')

def plot_verification_barplots(rate):
    """
    Plots average distribution bar plots directly from the verification CSV.
    Overlays the initial (t=0) distribution as hollow bars plotted manually.
    """
    print(f"\n--- Generating plots from VERIFICATION CSV for R = {rate} ---")
    verification_file = os.path.join(OUTPUT_DIR_TABLES, "VERIFICATION_final_step_agent_wealth.csv")
    
    try:
        df_final = pd.read_csv(verification_file)
    except FileNotFoundError:
        print(f"  > ERROR: Could not find {verification_file}.")
        return

    agg_agent_path = os.path.join(OUTPUT_DIR_PLOTS, "aggregated_agent_data_all_runs.csv")
    try:
        agent_data = pd.read_csv(agg_agent_path)
        df_initial = agent_data[(agent_data['Step'] == 0)].copy()
    except FileNotFoundError:
        print(f"  > ERROR: Could not find {agg_agent_path} for initial data.")
        return

    data_to_plot_final = df_final[df_final['Rate'] == rate].copy()
    data_to_plot_initial = df_initial[df_initial['Rate'] == rate].copy()
    
    global EXPERIMENTS_TO_PLOT 
    if EXPERIMENTS_TO_PLOT:
        print(f"  > VERIFICATION PLOT: Filtering for {EXPERIMENTS_TO_PLOT}")
        data_to_plot_final = data_to_plot_final[data_to_plot_final['Experiment'].isin(EXPERIMENTS_TO_PLOT)]
    
    if data_to_plot_final.empty:
        print(f"  > No data found in verification file for R = {rate} after filtering.")
        return

    # --- Replication Matching ---
    experiments = sorted(data_to_plot_final['Experiment'].unique())
    unique_replications = data_to_plot_final['Replication'].unique()
    matching_reps = None
    if 'baseline' in experiments:
        non_baseline_data = data_to_plot_final[data_to_plot_final['Experiment'] != 'baseline']
        if not non_baseline_data.empty:
            rep_counts = non_baseline_data.groupby('Experiment')['Replication'].nunique()
            min_reps_exp = rep_counts.idxmin()
            min_reps_n = rep_counts.min()
            matching_reps = non_baseline_data[non_baseline_data['Experiment'] == min_reps_exp]['Replication'].unique()
            print(f"  > VERIFICATION PLOT: Matching replications, N={min_reps_n} (from '{min_reps_exp}')")
            
            data_to_plot_final = data_to_plot_final[data_to_plot_final['Replication'].isin(matching_reps)]
            data_to_plot_initial = data_to_plot_initial[data_to_plot_initial['Replication'].isin(matching_reps)]
            unique_replications = matching_reps

    for log_scale in [True, False]:
        
        # 1. Clip data
        FLOOR_VALUE = 1e-2
        data_to_plot_final['Wealth'] = np.clip(data_to_plot_final['Wealth'], a_min=FLOOR_VALUE, a_max=None)
        data_to_plot_initial['Wealth'] = np.clip(data_to_plot_initial['Wealth'], a_min=FLOOR_VALUE, a_max=None)
        
        binned_data_all_reps = []
        binned_data_initial = []
        bin_order = []

        if log_scale:
            plot_title_suffix = '(Escala Log)'
            plot_filename_suffix = "LOG"
            
            bin_edges = np.power(10.0, np.arange(-2, 8)) # Changed from 7 to 8
            bin_labels = [f"$10^{{{i}}}$ a $10^{{{i+1}}}$" for i in range(-2, 7)] # Changed from 6 to 7
            
            bin_order = bin_labels
            
            # Bin FINAL data
            grouped_final = data_to_plot_final.groupby(['Experiment', 'Replication'])
            for (exp, rep), group_data in grouped_final:
                binned_wealth = pd.cut(group_data['Wealth'], bins=bin_edges, labels=bin_labels, right=False)
                bin_counts = binned_wealth.value_counts().reset_index()
                bin_counts.columns = ['Wealth Bin Label', 'Count']
                bin_counts['Experiment'] = exp
                bin_counts['Replication'] = rep
                binned_data_all_reps.append(bin_counts)
            
            # Bin INITIAL data
            grouped_initial = data_to_plot_initial.groupby('Replication')
            for rep, group_data in grouped_initial:
                binned_wealth = pd.cut(group_data['Wealth'], bins=bin_edges, labels=bin_labels, right=False)
                bin_counts = binned_wealth.value_counts().reset_index()
                bin_counts.columns = ['Wealth Bin Label', 'Count']
                bin_counts['Replication'] = rep
                binned_data_initial.append(bin_counts)
            
            if not binned_data_all_reps:
                print(f"  > No binned verification data to plot for R={rate}, log scale. Skipping.")
                continue
                
            binned_df = pd.concat(binned_data_all_reps, ignore_index=True)
            binned_df_initial = pd.concat(binned_data_initial, ignore_index=True) if binned_data_initial else pd.DataFrame()
        
        else: 
            plot_title_suffix = '(Escala Lineal)'
            plot_filename_suffix = "LINEAR"

            data_min = FLOOR_VALUE
            data_max = data_to_plot_final['Wealth'].max()
            if data_max <= data_min: data_max = data_min + 1

            num_bins = 25
            bins = np.linspace(data_min, data_max, num_bins)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            binned_data_final = []
            
            for exp in experiments:
                for rep in unique_replications:
                    rep_data_mask = (data_to_plot_final['Experiment'] == exp) & (data_to_plot_final['Replication'] == rep)
                    rep_wealth_data = data_to_plot_final.loc[rep_data_mask, 'Wealth']
                    
                    if not rep_wealth_data.empty:
                        counts, _ = np.histogram(rep_wealth_data, bins=bins)
                        for i in range(len(counts)):
                            binned_data_final.append({
                                'Experiment': exp,
                                'Replication': rep,
                                'Wealth Bin': bin_centers[i],
                                'Count': counts[i]
                            })
            
            for rep in unique_replications:
                rep_data_mask = data_to_plot_initial['Replication'] == rep
                rep_wealth_data = data_to_plot_initial.loc[rep_data_mask, 'Wealth']
                
                if not rep_wealth_data.empty:
                    counts, _ = np.histogram(rep_wealth_data, bins=bins)
                    for i in range(len(counts)):
                        binned_data_initial.append({
                            'Replication': rep,
                            'Wealth Bin': bin_centers[i],
                            'Count': counts[i]
                        })

            if not binned_data_final:
                print(f"  > No binned verification data to plot for R={rate}, linear scale. Skipping.")
                continue
                
            binned_df = pd.DataFrame(binned_data_final)
            binned_df['Wealth Bin Label'] = binned_df['Wealth Bin'].apply(lambda x: f'{x:.1e}')
            
            binned_df_initial = pd.DataFrame(binned_data_initial) if binned_data_initial else pd.DataFrame()
            if not binned_df_initial.empty:
                binned_df_initial['Wealth Bin Label'] = binned_df_initial['Wealth Bin'].apply(lambda x: f'{x:.1e}')
            
            bin_order = binned_df['Wealth Bin Label'].unique()


        plt.figure(figsize=(14, 8))
        palette_to_use = "tab10" if len(experiments) >= 5 else "viridis"
        ax = plt.gca() # Get current axes

        sns.barplot(data=binned_df, 
                     x='Wealth Bin Label', 
                     y='Count', 
                     hue='Experiment', 
                     order=bin_order,
                     palette=palette_to_use, 
                     ci='sd',
                     ax=ax)
                     
        if not binned_df_initial.empty:
            # Calculate initial means and std devs per bin 
            initial_stats = binned_df_initial.groupby('Wealth Bin Label')['Count'].agg(['mean', 'std']).reindex(bin_order)

            # Get the numeric locations of the x-ticks set by seaborn
            xticks_locs = ax.get_xticks()
            xticklabels = [label.get_text() for label in ax.get_xticklabels()]
            
            if len(xticks_locs) != len(bin_order):
                 print(f"Warning: [VERIFICATION] Mismatch between number of ticks ({len(xticks_locs)}) and number of bins ({len(bin_order)}). Initial distribution overlay might be incorrect.")
            else:
                label_to_loc = {label: loc for label, loc in zip(bin_order, xticks_locs)} 

                bar_width = 0.8 # Standard bar width
                for bin_label in bin_order:
                     if bin_label in initial_stats.index and bin_label in label_to_loc:
                         mean_val = initial_stats.loc[bin_label, 'mean']
                         std_val = initial_stats.loc[bin_label, 'std']
                         x_pos = label_to_loc[bin_label] # Get numeric position

                         # Plot the hollow bar
                         ax.bar(x_pos, mean_val, width=bar_width,
                                facecolor=(1,1,1,0), edgecolor='black', linewidth=1.0,
                                label='_nolegend_') # Avoid auto-labeling

                         # Plot the error bar
                         ax.errorbar(x_pos, mean_val, yerr=std_val, fmt='none',
                                     ecolor='black', elinewidth=1.0, capsize=3, label='_nolegend_')
                          
        plt.title(f'[VERIFICATION] Final Wealth Avg. Distribution {plot_title_suffix}, R = {rate}', fontsize=16)
        plt.xlabel('Riqueza', fontsize=12) # Your label
        plt.ylabel('Número de Agentes', fontsize=12) # Your label
        
        plt.xticks(rotation=45, ha='right', fontsize=9)
        
        handles, labels = ax.get_legend_handles_labels()
        if not binned_df_initial.empty and 'Initial (t=0)' not in labels:
             initial_patch = Patch(facecolor='none', edgecolor='black', linewidth=1.0, label='Initial (t=0)')
             handles.append(initial_patch)
             labels.append('Initial (t=0)')
        by_label = dict(zip(labels, handles)) 
        ax.legend(by_label.values(), by_label.keys(), title='Experiment')
        
        plt.tight_layout()
        
        filename = f"VERIFICATION_AvgDist_Barplot_{plot_filename_suffix}_R_{rate}.png"
        plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, filename))
        plt.close()
        print(f"  > Saved VERIFICATION Avg. Dist. Barplot ({plot_filename_suffix} scale) to: {filename}")

def perform_statistical_analysis(data, metric, rate):
    print(f"\n--- Análisis Estadístico (Welch + Games-Howell) para '{metric}' en R={rate} (Salida en Consola) ---")
    final_step_data = data[(data['Rate'] == rate) & (data['Step'] == SIMULATION_STEPS - 1)]
    experiments = final_step_data['Experiment'].unique()
    
    if len(experiments) < 2:
        print(f"  > Solo se encontró un grupo experimental. Omitiendo pruebas.")
        return
    if 'baseline' not in experiments:
        print("  > Experimento 'baseline' no encontrado. Omitiendo pruebas post-hoc.")
        return
        
    df_metric = final_step_data[['Experiment', metric]].copy().dropna()
    
    df_metric[metric] = df_metric[metric].astype(float) + np.random.uniform(-1e-4, 1e-4, size=len(df_metric))
    
    try:
        anova_res = pg.welch_anova(dv=metric, between='Experiment', data=df_metric)
        
        f_cols = [c for c in anova_res.columns if c.lower() in ['f', 'f-val', 'fval']]
        f_val = anova_res[f_cols[0]].iloc[0] if f_cols else 'N/A'
        
        p_cols = [c for c in anova_res.columns if c.lower() in ['p_unc', 'p-unc', 'p-val', 'pval', 'pvalue', 'p', 'pr(>f)']]
        if not p_cols:
            print(f"DEBUG: No se reconoció la columna P. Columnas de pingouin: {anova_res.columns.tolist()}")
            return
            
        p_val_anova = anova_res[p_cols[0]].iloc[0]
        
        f_val_str = f"{f_val:.4f}" if isinstance(f_val, (int, float)) else f_val
        print(f"Resultado ANOVA de Welch: Estadístico F = {f_val_str}, valor p = {p_val_anova:.4e}")
        
        if p_val_anova < 0.05:
            print("El ANOVA de Welch es significativo. Realizando post-hoc de Games-Howell contra el baseline...")
            posthoc = pg.pairwise_gameshowell(dv=metric, between='Experiment', data=df_metric)
            
            for exp in experiments:
                if exp != 'baseline':
                    match = posthoc[((posthoc['A'] == exp) & (posthoc['B'] == 'baseline')) | 
                                    ((posthoc['A'] == 'baseline') & (posthoc['B'] == exp))]
                    if not match.empty:
                        p_cols_gh = [c for c in match.columns if c.lower() in ['p_unc', 'p-unc', 'pval', 'p-val', 'pvalue', 'p']]
                        if p_cols_gh:
                            p_val_gh = match[p_cols_gh[0]].iloc[0]
                            significance = "SIGNIFICATIVO" if p_val_gh < 0.05 else "no significativo"
                            print(f"  - Games-Howell '{exp}' vs 'baseline': valor p = {p_val_gh:.4e} ({significance})")
        else:
            print("El ANOVA de Welch no es significativo. No proceden pruebas post-hoc.")
            
    except Exception as e:
        print(f"Error realizando el análisis estadístico para {metric}: {e}")


# --- 5. MAIN EXECUTION BLOCK ---

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
    try:
        print(f"\n--- Exporting final step wealth for verification... ---")

        final_step_number = SIMULATION_STEPS - 1 
        
       
        final_wealth_df = agent_data[agent_data['Step'] == final_step_number].copy()
        
       
        verification_path = os.path.join(OUTPUT_DIR_TABLES, "VERIFICATION_final_step_agent_wealth.csv")
        
        
        columns_to_save = ['Experiment', 'Rate', 'Replication', 'AgentID', 'Wealth']
        
   
        final_wealth_df_to_save = final_wealth_df[columns_to_save]
        final_wealth_df_to_save.to_csv(verification_path, index=False, float_format='%.5f')
        
        print(f"  > Successfully saved verification file to: {verification_path}")

    except Exception as e:
        print(f"  > FAILED to export verification file. Error: {e}")
        print("  > This might be because 'AgentID' is still not found.")
        print("  > Make sure you deleted the old cache files first.")
    

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



    all_available_rates = sorted(model_data['Rate'].unique())
    
    rates_to_iterate = []
    if not RATES_TO_PROCESS:
        rates_to_iterate = all_available_rates
        print(f"\n--- No specific rates selected. Analyzing all found rates: {rates_to_iterate} ---")
    else:
        rates_to_iterate = [r for r in RATES_TO_PROCESS if r in all_available_rates]
        
        missing_rates = [r for r in RATES_TO_PROCESS if r not in all_available_rates]
        if missing_rates:
            print(f"\n--- Warning: Could not find data for specified rates: {missing_rates} ---")
        
        print(f"\n--- Analyzing user-specified rates: {rates_to_iterate} ---")

    if not rates_to_iterate:
            print("\n--- No data found for any of the specified rates. Exiting analysis. ---")


    for interest_rate in rates_to_iterate:
    
        print(f"\n{'='*25} ANALYZING RESULTS FOR R = {interest_rate} {'='*25}")
        
        rate_model_data = model_data[model_data['Rate'] == interest_rate].copy()
        rate_agent_data = agent_data[agent_data['Rate'] == interest_rate].copy()

        # Check if there is data after filtering for the rate
        if rate_model_data.empty or rate_agent_data.empty:
            print(f"  > No data found for R = {interest_rate}. Skipping analysis for this rate.")
            continue

        generate_summary_table(rate_model_data, rate_agent_data, interest_rate) 
        generate_statistical_table(rate_model_data, rate_agent_data, interest_rate)
        
        print(f"\n--- Ejecutando análisis estadístico en consola para R = {interest_rate} ---")
        model_metrics_to_plot = ["Average Wealth", "Gini_Coefficient", "Average Consumption", "Average Savings"]
        for metric in model_metrics_to_plot:
            perform_statistical_analysis(rate_model_data, metric, interest_rate)


        if GENERATE_PLOTS:
            print(f"\n--- Generando Paneles Compuestos para R = {interest_rate} ---")
            
            plot_composite_time_series(rate_model_data, interest_rate)
            plot_composite_distributions(rate_agent_data, interest_rate)

            print(f"--- Generando gráficas por perfil para R = {interest_rate} ---")
            plot_wealth_by_profile(rate_agent_data, interest_rate)
            plot_beta_by_profile(rate_agent_data, interest_rate)
            
            plot_verification_barplots(interest_rate) 
            
        else:
            print(f"\n--- Omitiendo generación de gráficas para R = {interest_rate} (GENERATE_PLOTS = False) ---")
    print("\n========================= ANALYSIS COMPLETE =========================")