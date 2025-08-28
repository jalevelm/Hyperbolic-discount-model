import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# --- 1. Setup ---
output_dir_csv = "output_csv"
output_dir_plots = "output_plots"
os.makedirs(output_dir_plots, exist_ok=True)

all_files = os.listdir(output_dir_csv)
agent_data_files = sorted([f for f in all_files if "agent_data" in f])
model_data_files = sorted([f for f in all_files if "model_data" in f])

print(f"Found {len(agent_data_files)} agent data files and {len(model_data_files)} model data files.")

# --- 2. Generate Plots for Each Experimental Run ---

for model_file, agent_file in zip(model_data_files, agent_data_files):
    print(f"\n--- Processing: {model_file} ---")

    # --- Load the Data ---
    model_data = pd.read_csv(os.path.join(output_dir_csv, model_file), index_col=0)
    agent_data = pd.read_csv(os.path.join(output_dir_csv, agent_file), index_col=[0, 1])
    
    try:
        rate = float(model_file.split('_')[1])
    except:
        rate = 'Unknown'

    # --- Plot 1: High-Level Aggregate Behavior (Now with 4 Subplots) ---
    fig, axes = plt.subplots(4, 1, figsize=(12, 22), sharex=True)
    fig.suptitle(f'Aggregate Model Behavior (R = {rate})', fontsize=16)

    model_data["Average Wealth"].plot(ax=axes[0], title="Average Wealth", grid=True)
    axes[0].set_ylabel("Wealth")

    model_data["Average Consumption"].plot(ax=axes[1], title="Average Consumption", grid=True)
    axes[1].set_ylabel("Consumption")
    
    model_data["Average Savings"].plot(ax=axes[2], title="Average Savings", grid=True)
    axes[2].set_ylabel("Savings")

    model_data["Average Utility"].plot(ax=axes[3], title="Average Utility", grid=True)
    axes[3].set_ylabel("Utility")
    axes[3].set_xlabel("Step")

    plot_path = os.path.join(output_dir_plots, f"aggregate_metrics_R_{rate}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"  > Saved aggregate plot: {plot_path}")

    if "Gini_Coefficient" in model_data.columns and "Wealth_Quantile_90" in model_data.columns:
        fig, axes = plt.subplots(2, 1, figsize=(12, 14), sharex=True)
        fig.suptitle(f'Wealth Distribution and Inequality (R = {rate})', fontsize=16)

        # Subplot 1: Gini Coefficient
        model_data["Gini_Coefficient"].plot(ax=axes[0], title="Gini Coefficient Over Time", grid=True, color='red')
        axes[0].set_ylabel("Gini Coefficient (0 = Equality)")
        axes[0].set_ylim(0, 1) # Gini is always between 0 and 1

        # Subplot 2: Wealth Quantiles
        model_data["Wealth_Quantile_10"].plot(ax=axes[1], title="Wealth Quantiles", grid=True, label='10th Percentile (Bottom 10%)')
        model_data["Wealth_Quantile_90"].plot(ax=axes[1], label='90th Percentile (Top 10%)')
        axes[1].set_ylabel("Wealth Level")
        axes[1].set_xlabel("Step")
        axes[1].legend()
        axes[1].set_yscale('log') # Use a log scale if the wealth gap is very large
        axes[1].set_title("Wealth Gap: Top 10% vs. Bottom 10%")


        plot_path = os.path.join(output_dir_plots, f"inequality_metrics_R_{rate}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"  > Saved inequality plot: {plot_path}")

    # --- Plot 2: "Zoom-In" on Individual Agents ---
    fig, axes = plt.subplots(4, 1, figsize=(12, 22), sharex=True)
    fig.suptitle(f'Individual Agent Metrics (R = {rate})', fontsize=16)

    profiles = agent_data['Profile'].unique()
    agents_to_plot = {}
    for profile in profiles:
        
        agent_id = agent_data[agent_data['Profile'] == profile].index.get_level_values('AgentID')[0]
        agents_to_plot[profile] = agent_id

    for profile, agent_id in agents_to_plot.items():
        agent_specific_data = agent_data.loc[(slice(None), agent_id), :]
        steps = agent_specific_data.index.get_level_values('Step')
        
        axes[0].plot(steps, agent_specific_data.Wealth, label=f'Agent {agent_id} ({profile})')
        axes[1].plot(steps, agent_specific_data.Consumption, label=f'Agent {agent_id} ({profile})')
        axes[2].plot(steps, agent_specific_data.Utility, label=f'Agent {agent_id} ({profile})')
        axes[3].plot(steps, agent_specific_data.Policy_Savings, label=f'Plan (Agent {agent_id})', linestyle='--', alpha=0.8)
        axes[3].plot(steps, agent_specific_data.Socially_Adjusted_Goal, label=f'Intent (Agent {agent_id})', linestyle=':', alpha=0.8)
        axes[3].plot(steps, agent_specific_data.Savings, label=f'Action (Agent {agent_id})', linestyle='-', alpha=1.0)

    axes[0].set_title("Wealth Over Time"); axes[0].set_ylabel("Wealth"); axes[0].legend(); axes[0].grid(True)
    axes[1].set_title("Consumption Over Time"); axes[1].set_ylabel("Consumption"); axes[1].legend(); axes[1].grid(True)
    axes[2].set_title("Utility Over Time"); axes[2].set_ylabel("Utility"); axes[2].set_xlabel("Step"); axes[2].legend(); axes[2].grid(True)
    axes[3].set_title("Savings: Plan vs. Intent vs. Action"); axes[3].set_ylabel("Savings"); axes[3].set_xlabel("Step"); axes[3].legend(); axes[3].grid(True)
    
    plot_path = os.path.join(output_dir_plots, f"individual_metrics_R_{rate}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"  > Saved individual agent plot: {plot_path}")
    
    # --- Plot 3 & 4: V and g functions ---
    output_dir_v_g = os.path.join(output_dir_csv, "v_g_functions")
    
    # Find a sample agent ID for each profile to display in the legend
    profiles = agent_data['Profile'].unique()
    sample_agent_ids = {}
    for profile in profiles:
        first_agent_id = agent_data[agent_data['Profile'] == profile].index.get_level_values('AgentID')[0]
        sample_agent_ids[profile] = first_agent_id

    # The wealth grid needs to match the one used in the simulation
    wealth_grid = np.geomspace(1e-6, 1000001, 300) 
    
    rate_str = f"R_{rate}_"
    npy_files = [f for f in os.listdir(output_dir_v_g) if f.startswith(rate_str)]
    
    policy_files = sorted([f for f in npy_files if "policy_function" in f])
    if policy_files:
        plt.figure(figsize=(12, 7))
        for g_file in policy_files:
            g_func = np.load(os.path.join(output_dir_v_g, g_file))
            # --- Corrected Label Parsing ---
            parts = g_file.split('_') # e.g., ['R', '1.3', 'planner', 'policy', 'function.npy']
            profile = parts[2]
            agent_id = sample_agent_ids.get(profile, 'N/A') # Get a sample ID for the legend
            label = f'Agent {agent_id} ({profile})'
            plt.plot(wealth_grid, g_func, label=label)

        plt.plot(wealth_grid, wealth_grid, 'k--', label='k\' = k (No Change)', alpha=0.7)
        plt.title(f'Agent Policy Functions g(k) (R = {rate})')
        plt.xlabel('Current Wealth (k)'); plt.ylabel("Next Period's Wealth / Savings (k')")
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(output_dir_plots, f"policy_functions_R_{rate}.png")); plt.close()
        print(f"  > Saved policy function plot.")

    value_files = sorted([f for f in npy_files if "value_function" in f])
    if value_files:
        plt.figure(figsize=(12, 7))
        for v_file in value_files:
            v_func = np.load(os.path.join(output_dir_v_g, v_file))
            # --- Corrected Label Parsing ---
            parts = v_file.split('_') # e.g., ['R', '1.3', 'planner', 'value', 'function.npy']
            profile = parts[2]
            agent_id = sample_agent_ids.get(profile, 'N/A') # Get a sample ID for the legend
            label = f'Agent {agent_id} ({profile})'
            plt.plot(wealth_grid, v_func, label=label)

        plt.title(f'Agent Value Functions V(k) (R = {rate})'); plt.xlabel('Current Wealth (k)'); plt.ylabel('Value V(k)')
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(output_dir_plots, f"value_functions_R_{rate}.png")); plt.close()
        print(f"  > Saved value function plot.")

    # --- Plot 5: Individual Agent Resource Allocation (Stacked Bar Chart) ---
    
    # We use the same 'agents_to_plot' dictionary from Plot 2
    for profile, agent_id in agents_to_plot.items():
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Filter the dataframe for this specific agent
        agent_specific_data = agent_data.loc[(slice(None), agent_id), :].reset_index()

        # We need to exclude the initial state at Step 0 for this plot, as no decision was made yet
        plot_data = agent_specific_data[agent_specific_data['Step'] > 0]
        
        steps = plot_data['Step']
        consumption = plot_data['Consumption']
        savings = plot_data['Savings']
        total_resources = plot_data['Total Resources']

        # Create the stacked bar chart
        ax.bar(steps, consumption, label='Consumption', color='skyblue')
        ax.bar(steps, savings, bottom=consumption, label='Savings (Next Period Wealth)', color='salmon')

        # Optionally, plot the total resources line to show where the bars should reach
        ax.plot(steps, total_resources, color='black', linestyle='--', marker='o', label='Total Resources Available')

        ax.set_title(f'Agent {agent_id} ({profile}) - Resource Allocation per Step (R = {rate})')
        ax.set_xlabel('Step')
        ax.set_ylabel('Amount')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plot_path = os.path.join(output_dir_plots, f"allocation_agent_{agent_id}_{profile}_R_{rate}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"  > Saved resource allocation plot: {plot_path}")

    # --- Plot 6: Beta Convergence Over Time ---
    
    # We use the same 'agents_to_plot' dictionary from Plot 2 for consistency
    fig, ax = plt.subplots(figsize=(12, 7))

    for profile, agent_id in agents_to_plot.items():
        # Filter the dataframe for each specific agent
        agent_specific_data = agent_data.loc[(slice(None), agent_id), :]
        steps = agent_specific_data.index.get_level_values('Step')
        
        # Plot the Beta value over the steps
        ax.plot(steps, agent_specific_data.Beta, label=f'Agent {agent_id} ({profile})', marker='.', markersize=4)

    # For context, calculate the initial average beta of the entire population
    
    if 1 in agent_data.index.get_level_values('Step'):
        initial_mean_beta = agent_data.loc[1]['Beta'].mean()
        ax.axhline(y=initial_mean_beta, color='k', linestyle='--', 
                    label=f'Initial Mean Beta (~{initial_mean_beta:.3f})')

    ax.set_title(f'Agent Beta Convergence Over Time (R = {rate})')
    ax.set_xlabel('Step')
    ax.set_ylabel('Beta (Present Bias Parameter)')
    ax.legend()
    ax.grid(True)
    
    plot_path = os.path.join(output_dir_plots, f"beta_convergence_R_{rate}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"  > Saved beta convergence plot: {plot_path}")


print("\n--- Analysis complete. All plots saved to 'output_plots' directory. ---")