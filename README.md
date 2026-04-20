```
# Saving and Dissaving with Hyperbolic Discounting - Agent-Based Model

This repository contains an Agent-Based Model (ABM) simulating saving and dissaving behavior under hyperbolic discounting. Inspired by and implementing concepts from the paper "Saving and Dissaving with Hyperbolic Discounting" by Cao and Werning (2018), this model expands the theoretical framework by introducing social dynamics and heterogeneous agent profiles in a network setting.

## Model Overview

The model simulates a population of agents making intertemporal consumption and saving decisions over a finite horizon. Agents exhibit **time-inconsistent (hyperbolic) preferences** characterized by:
- `beta` ($\beta$): Present bias parameter. Values < 1 indicate a preference for immediate gratification.
- `delta` ($\delta$): Standard exponential discount factor.

To bridge theoretical economic models with sociological realism, agents do not act in isolation. They are embedded within a social network (e.g., Watts-Strogatz small-world network) and influence one another through various configurable mechanisms.

## Key Features

- **Value Function Iteration (VFI):** Solves the rational benchmark for each agent profile using parallelized numerical optimization to approximate Markov perfect equilibria. The results are cached to ensure computational efficiency during the simulation.
- **Agent Heterogeneity:** The population is composed of distinct behavioral profiles (e.g., *Planners*, *Moderates*, *Procrastinators*, *Impulsive*, and *Inverse Procrastinators*), each with distinct $\beta$, $\delta$, and financial literacy levels.
- **Social Mechanisms:**
  - **Information Diffusion:** Agents can learn from neighbors with higher financial literacy, gradually updating their own literacy and shifting their decisions toward the "planner" optimal policy.
  - **Peer Comparison:** Agents observe the consumption of their peers and adjust their savings goals to maintain a socially acceptable level of consumption.
  - **Social Norms:** Agents' fundamental preferences ($\beta$) slowly adapt toward the prevailing norm within their social neighborhood.
- **Comprehensive Output & Analysis:** Includes a robust analysis pipeline to generate comprehensive summary tables, statistical significance tests (Welch's ANOVA and Games-Howell post-hoc), and APA-formatted visualizations of wealth trajectories, distributions, and policy functions.

## Prerequisites

The project requires Python 3.8+ and several external libraries. To set up the environment, install the  dependencies contained in the requirements.txt file:


## Replication Instructions

Replicating the model and its findings is divided into two primary phases: generating the simulation data and analyzing the results.

### 1. Running the Simulation

Execute the main model script to run the simulations. The script will automatically pre-compute the Value Functions (if not cached), simulate the defined experiments, and export the data.

```bash
python Model.py
```

**What happens during execution?**
- The script iterates through defined experimental setups (e.g., `baseline`, `peer_comparison_only`, `social_norms_only`, `info_diffusion_only`, `all_interactions`).
- It outputs raw execution logs to `output_text/simulation_run_log.txt`.
- Agent-level and Model-level data are saved as `.csv` files in the `output_csv/` directory.
- Calculated Value ($V$) and Policy ($g$) functions are saved as `.npy` arrays in `output_csv/v_g_functions/`.

### 2. Analyzing the Results

Once the simulation completes, run the analysis script to parse the output and generate plots and tables.

```bash
python analyze_results.py
```

**What happens during execution?**
- Aggregates all replication data.
- Evaluates economic metrics such as the Gini Coefficient, Wealth Quantiles, Average Savings, and Consumption.
- Outputs summary and statistical tables (CSV format) to the `output_tables/` directory.
- Generates high-quality visualizations (time series, wealth distributions, and VFI policy curves) in the `output_plots/` directory.

## Project Structure

- `Model.py`: The core ABM implementation. Contains the `SavingAgent` class (agent logic and decision-making), the `SavingModel` class (environment setup, network generation, and scheduling), and the simulation runner.
- `analyze_results.py`: The data processing, statistical testing, and visualization pipeline.
- `output_csv/`: Directory containing raw simulation data (created automatically).
- `output_plots/`: Directory containing all generated charts and graphs (created automatically).
- `output_tables/`: Directory containing calculated metrics and statistical test results (created automatically).

## Customizing the Experiment

You can easily modify the simulation conditions by editing the configuration blocks at the bottom of `Model.py`:

- **Agent Profiles:** Modify the `agent_profiles` dictionary to change $\beta$, $\delta$, or financial literacy.
- **Population:** Adjust the `population_to_simulate` dictionary to change the number of agents per profile.
- **Economic Environment:** Change the `interest_rates_to_test` list or adjust the initial `wealth_dist`.
- **Experiments:** Toggle social mechanisms in the `experiments` dictionary.

Similarly, in `analyze_results.py`, you can modify constants like `EXPERIMENTS_TO_PLOT` or `RATES_TO_PROCESS` to focus the output generation on specific scenarios.

## References

- Cao, D., & Werning, I. (2018). Saving and Dissaving with Hyperbolic Discounting. *Econometrica*, 86(3), 805-857.
