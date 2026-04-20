# Model.py - Core ABM implementation for saving and dissaving with hyperbolic discounting
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

import sys 
import os 
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid, NetworkGrid
from mesa.datacollection import DataCollector  
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize  
from joblib import Parallel, delayed
import pandas as pd
import cProfile
import pstats
import math
import networkx as nx

class SavingAgent(Agent):
    """
    Represents an agent with time-inconsistent (hyperbolic) preferences
    making intertemporal consumption and saving decisions.  This agent
    implements a simplified version of the Cao-Werning (2018) model.

    Attributes:
        unique_id (int): A unique identifier for the agent.
        model (Model): A reference to the model the agent belongs to.
        beta (float): The present bias parameter.  Values < 1 indicate a
            preference for immediate consumption over future consumption.
        delta (float): The discount factor. Represents the weight placed on
            future utility relative to current utility.
        init_wealth (float): The agent's initial wealth level.
        sigma (float): inverse of the elasticity of intertemporal substitution (IES=1/σ), 
            which governs the willingness of agents to substitute consumption between 
            different time periods based on changes in the interest rate
        borrowing_limit (float): The minimum allowed wealth level (can be
            negative to allow borrowing).  Set to 0 in this implementation.
        savings (float): The amount the agent chooses to save in the current
            period. Initialized to 0.
        R_star (float): The threshold interest rate.  The agent theoretically
            saves if R > R_star and dissaves if R < R_star in all Markov
            equilibria. Calculated based on beta and delta.
        previous_savings (float): The agent's savings in the previous time step.
            Initialized to None. Used for tracking.
        init_wealth (float): Stores the initial wealth of the agent.
        step_count (int): Tracks the number of steps (time periods) the agent
            has been active. Initialized to 0.
        previous_wealth (float): Stores the wealth of the agent at the
            beginning of the current step. Initialized to init_wealth.
        value_function_calculated (bool): A flag indicating whether the value
            function (V) and policy function (g) have been calculated via
            value iteration.  Initialized to False.
        V (np.ndarray):  The value function, representing the maximized
            discounted lifetime utility attainable from each possible wealth
            level.  Initialized to None, calculated during the first step.
        g (np.ndarray): The policy function, indicating the optimal next-period
            wealth level (savings) for each possible current wealth level.
            Initialized to None, calculated during the first step.
    """
    def __init__(self, unique_id, model, profile_name, beta, delta, init_wealth, sigma, borrowing_limit, financial_literacy, iterations=50):
        """
        Initializes a new SavingAgent.

        Args:
            unique_id (int): Unique identifier for the agent.
            model (Model): Reference to the simulation model.
            beta (float): Present bias parameter (0 < beta <= 1).
            delta (float): Discount factor (0 < delta < 1).
            init_wealth (float): Initial wealth of the agent.
            sigma (float): Coefficient of relative risk aversion.
            borrowing_limit (float): Minimum allowed wealth level.
        """
        super().__init__(unique_id, model)  

        self.profile_name = profile_name
        self.original_profile = profile_name
        # --- Agent Preferences and State ---
        self.beta = beta        # Present bias
        self.delta = delta      # Discount factor
        self.sigma = sigma      # Inverse of elasticity of intertemporal substitution
        self.financial_literacy = financial_literacy

        # --- Initial Conditions ---
        self.wealth = init_wealth   # Current wealth
        self.init_wealth = init_wealth # Initial wealth (stored for reference)
        self.borrowing_limit = borrowing_limit  # Borrowing constraint
        self.consumption = 0
        self.savings = 0        # Initial savings (updated each step)
        self.policy_savings = 0                 
        self.socially_adjusted_savings_goal = 0
        self.previous_savings = None    # Savings from the previous step
        self.previous_wealth = init_wealth  # Wealth at the start of the step
        self.max_vfi_iterations = iterations
        
        # --- Derived Parameters ---
        self.R_star = 1 + (1 - self.delta) / (self.beta * self.delta)   # Threshold interest rate
        
        # --- Value and Policy Function (for Markov Equilibrium) ---
        self.value_function_calculated = False  # Flag: Value iteration performed?
        self.V = None  # Value function (array, calculated later)
        self.g = None  # Policy function (array, calculated later)

        # --- Internal Tracking ---
        self.step_count = 0     # Number of steps taken

        self.V_snapshots = {}
        self.g_snapshots = {}
        self.diff_V_history = [] 
        self.diff_g_history = [] 
        self.consumption_history = []
        self.wealth_history = [self.init_wealth] # Start with initial wealth for time step 0
        self.utility_history = []

    def utility(self, consumption):
        """
        Calculates the agent's utility from consumption in a single period.
        """
        epsilon = 1e-9
        consumption = max(consumption, epsilon)

        if self.sigma == 1:
            return np.log(consumption)
        else:
            return consumption**(1 - self.sigma) / (1 - self.sigma)

    def get_neighbors(self):
        """
        Returns a list of neighboring agent objects.
        The definition of "neighbor" depends on the grid type used.
        """
        if isinstance(self.model.grid, MultiGrid):
            # For a grid, neighbors are in adjacent cells. moore=True includes diagonals.
            return self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        elif isinstance(self.model.grid, NetworkGrid):
            # For a network, the NetworkGrid's get_neighbors() method 
            # returns the actual agent objects directly.
            return self.model.grid.get_neighbors(self.unique_id, include_center=False)
        return []

    def step(self):
        """
        Advances the agent by one time step, printing detailed logs of its actions.
        """
        if self.step_count == 0:
            print(f"Agent {self.unique_id} ({self.profile_name}): First Step - Beta: {self.beta}, Delta: {self.delta}, R_star: {self.R_star}")
            neighbors = self.get_neighbors()
            print(f"Agent {self.unique_id}: Has {len(neighbors)} neighbors.")
        
        print(f"Agent {self.unique_id} ({self.profile_name}): Step start. Wealth: {self.wealth:.4f}, Previous Savings: {self.previous_savings}")
        print(f"Agent {self.unique_id}: R = {self.model.interest_rate}, R_star = {self.R_star}")

        try:
            self.V, self.g = self.model.vfi_cache[self.profile_name]
        except KeyError:
            raise Exception(f"CRITICAL ERROR: VFI for profile '{self.profile_name}' not found in cache for Agent {self.unique_id}.")

        if self.model.information_diffusion_active and self.model.information_diffusion_strength > 0:
            neighbors = self.get_neighbors()
            if neighbors:
                # 1. Identify neighbors with higher financial literacy
                more_literate_neighbors = [n for n in neighbors if n.financial_literacy > self.financial_literacy]
                
                if more_literate_neighbors:
                    # 2. Calculate the average literacy of this more knowledgeable group
                    avg_smarter_literacy = np.mean([n.financial_literacy for n in more_literate_neighbors])
                    
                    # 3. Update the agent's own literacy, moving it towards the average
                    #    of the more literate neighbors. The strength parameter controls the speed.
                    learning_rate = self.model.information_diffusion_strength
                    self.financial_literacy += learning_rate * (avg_smarter_literacy - self.financial_literacy)

        # --- Agent's Decision (Every Step) ---
        wealth_grid = self.model.wealth_grid
        wealth_index = np.argmin(np.abs(wealth_grid - self.wealth))


        optimal_savings_from_policy = self.g[wealth_index]
        optimal_savings = optimal_savings_from_policy

        if self.model.information_diffusion_active:
            # 2. Get the "rational" savings plan from the planner's policy function
            _, g_planner = self.model.vfi_cache["planner"]
            rational_savings_goal = g_planner[wealth_index]
            
            # 3. Blend the agent's biased plan with the rational plan using financial_literacy as the weight
            # An agent with literacy=1 will fully adopt the planner's goal.
            # An agent with literacy=0 will stick to their own biased goal.
            optimal_savings = ((1 - self.financial_literacy) * optimal_savings_from_policy) + \
                              (self.financial_literacy * rational_savings_goal)

        if self.model.peer_comparison_active and self.model.peer_comparison_strength > 0:
            neighbors = self.get_neighbors()
            if neighbors:
                # 1. Observe neighbors' consumption from the previous step
                neighbor_consumptions = [n.consumption for n in neighbors]
                
                if neighbor_consumptions:
                    # 2. Calculate the social "signal" (average consumption)
                    avg_neighbor_consumption = np.mean(neighbor_consumptions)
                    
                    # 3. Determine the agent's own ideal consumption
                    ideal_consumption = max(self.model.interest_rate * self.wealth - optimal_savings, 1e-9)

                    # 4. Blend the ideal consumption with the social signal
                    strength = self.model.peer_comparison_strength
                    final_consumption = ((1 - strength) * ideal_consumption) + (strength * avg_neighbor_consumption)

                    # 5. The agent's new savings goal is based on this socially-adjusted consumption
                    optimal_savings = self.model.interest_rate * self.wealth - final_consumption

        if self.model.social_norm_active and self.model.social_norm_strength > 0:
            neighbors = self.get_neighbors()
            if neighbors:
                # 1. Observe the 'beta' parameter of neighbors to gauge the social norm
                neighbor_betas = [n.beta for n in neighbors]
                
                if neighbor_betas:
                    # 2. Calculate the prevailing norm (average beta in the neighborhood)
                    neighborhood_beta_norm = np.mean(neighbor_betas)
                    
                    # 3. Adjust the agent's own beta towards the norm
                    # This represents a slow change in the agent's preferences 
                    strength = self.model.social_norm_strength
                    self.beta = ((1 - strength) * self.beta) + (strength * neighborhood_beta_norm)

        self.policy_savings = optimal_savings_from_policy
        self.socially_adjusted_savings_goal = optimal_savings

        self.savings = np.clip(optimal_savings, self.borrowing_limit, self.model.interest_rate * self.wealth)
        
        wealth_with_interest = self.model.interest_rate * self.wealth


        self.consumption = max(wealth_with_interest - self.savings, 1e-9)
        percentage_consumed = (self.consumption / (wealth_with_interest + 1e-9)) * 100
        percentage_saved = (self.savings / (wealth_with_interest + 1e-9)) * 100

        print(f"Agent {self.unique_id}: Step {self.step_count}")
        print(f"  Previous Wealth: {self.previous_wealth:.4f}")
        print(f"  Optimal Savings (from policy): {optimal_savings_from_policy:.4f}")
        if abs(optimal_savings - optimal_savings_from_policy) > 1e-6:
             print(f"  Socially-Adjusted Savings Goal: {optimal_savings:.4f}")
        print(f"  Calculated Savings (after clipping): {self.savings:.4f}")
        print(f"  Consumption: {self.consumption:.4f}")
        
        # --- State Updates ---
        self.previous_wealth = self.wealth
        self.wealth = self.savings
        self.previous_savings = self.savings

        print(f"Agent {self.unique_id}: Step end. | Total resources: {wealth_with_interest:.2f}, Consumption: {self.consumption:.2f}, Savings: {self.savings:.2f}, Wealth end of step: {self.wealth:.2f}")
        print(f"Percentage consumed: {percentage_consumed:.2f}%, percentage saved: {percentage_saved:.2f}%\n")

        self.consumption_history.append(self.consumption)
        self.wealth_history.append(self.wealth)
        self.utility_history.append(self.utility(self.consumption))
        self.step_count += 1


def compute_gini(model):
    """Calculates the Gini coefficient of agent wealth."""
    agent_wealths = [agent.wealth for agent in model.schedule.agents]

    if len(agent_wealths) < 2:
        return 0
    # Formula for Gini coefficient
    x = sorted(agent_wealths)
    n = len(x)
    cumx = np.cumsum(x, dtype=float)
    # The Gini coefficient is the area between the Lorenz curve and the line of equality
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

def get_wealth_quantile(model, quantile):
    """Calculates the wealth at a given quantile."""
    agent_wealths = [agent.wealth for agent in model.schedule.agents]
    if not agent_wealths:
        return 0
    return np.quantile(agent_wealths, q=quantile)

# --- VFI Function---
def calculate_vfi_for_profile(profile_params, model_params):
    """
    Performs VFI and writes its progress to a log file.
    """
    log_path = profile_params['log_path']
    profile_name = profile_params['name']
    
    try:
        # Redirect all print statements within this block to the log file
        with open(log_path, 'w') as log_file:
            sys.stdout = log_file

            beta = profile_params['beta']
            delta = profile_params['delta']
            max_iterations = profile_params['vfi_iterations']
            R = model_params['interest_rate']
            sigma = model_params['sigma']
            wealth_grid = model_params['wealth_grid']
            borrowing_limit = model_params['borrowing_limit']
            R_star = 1 + (1 - delta) / (beta * delta)
            
            print(f"--- VFI Log for Profile: '{profile_name}' ---")
            print(f"Parameters: Beta={beta}, Delta={delta}, R={R}, R_star={R_star}\n")
            
            start_time = time.time()

            def utility(consumption):
                consumption = max(consumption, 1e-9)
                if sigma == 1: return np.log(consumption)
                else: return consumption**(1 - sigma) / (1 - sigma)

            def optimize_savings(k, V_old, grid):
                max_obj_val = -np.inf
                optimal_k_next = np.clip(k, borrowing_limit, R * k)
                for k_next_candidate in grid:
                    if not (borrowing_limit <= k_next_candidate <= R * k): continue
                    consumption = R * k - k_next_candidate
                    if consumption > 1e-9:
                        val_at_k_next = np.interp(k_next_candidate, grid, V_old)
                        obj_val = utility(consumption) + beta * delta * val_at_k_next
                        if obj_val > max_obj_val:
                            max_obj_val = obj_val
                            optimal_k_next = k_next_candidate
                continuation_val = np.interp(optimal_k_next, grid, V_old)
                return continuation_val, optimal_k_next

            V = np.zeros_like(wealth_grid)
            g = np.zeros_like(wealth_grid)
            tolerance = 1e-6
            
            for i in range(max_iterations):
                V_old = V.copy()
                g_old_this_iter = g.copy()
                
                for j, k_val in enumerate(wealth_grid):
                    continuation, next_k = optimize_savings(k_val, V_old, wealth_grid)
                    V[j] = utility(max(R * k_val - next_k, 1e-9)) + delta * continuation
                    g[j] = next_k
                
                diff_V = np.max(np.abs(V - V_old))
                diff_g_array = np.abs(g - g_old_this_iter)
                diff_g = np.max(diff_g_array)
                k_val_max_diff_g = np.nan
                if diff_g > 0:
                    k_val_max_diff_g = wealth_grid[np.argmax(diff_g_array)]

                if (i + 1) % 20 == 0:
                    print(f"VFI Iteration {(i+1)}/{max_iterations}, Diff_V: {diff_V:.4e}, Diff_g: {diff_g:.4e} (at k={k_val_max_diff_g:.2f})")

                if diff_V < tolerance:
                    total_time = time.time() - start_time
                    print(f"\nValue function CONVERGED after {i+1} iterations.")
                    print(f"Total value iteration took {total_time:.4f} seconds.")
                    # Return results to the main process
                    return (profile_name, V, g)
                    
            total_time = time.time() - start_time
            print(f"\nValue function DID NOT CONVERGE after {max_iterations} iterations.")
            print(f"Total value iteration took {total_time:.4f} seconds.")
            # Return results to the main process
            return (profile_name, V, g)
    finally:
        # Restore standard output for the worker process
        sys.stdout = original_stdout

class SavingModel(Model):
    """
    A Mesa model simulating the saving behavior of agents with
    time-inconsistent preferences, based on the Cao-Werning (2018) model.

    The model features  agents that make
    consumption and savings decisions in discrete time over a finite horizon.
    Each agent has hyperbolic discounting preferences and faces a constant
    interest rate and a borrowing constraint.
    """
    
    def __init__(self, population_composition, interest_rate, sigma, wealth_dist, 
                 num_wealth_points=100, network='grid', network_params=None, 
                 seed=None,
                 peer_comparison_active=False,
                 social_norm_active=False,
                 information_diffusion_active=False,
                 peer_comparison_strength=0.05,
                 social_norm_strength=0.05,
                 information_diffusion_strength=0.05,
                 vfi_recalculation_interval=20,
                 vfi_cache=None):

        super().__init__(seed=seed)
        np.random.seed(seed)    

        if vfi_cache:
            print("--- Loading pre-computed VFI cache. ---")
            self.vfi_cache = vfi_cache
        else:
            print("--- No cache provided. Starting VFI Pre-computation for all profiles ---")
            start_time = time.time()
            self.vfi_cache = {}

            unique_profiles_to_compute = population_composition.keys()
            vfi_log_dir = os.path.join(output_dir_text, "vfi_logs")
            os.makedirs(vfi_log_dir, exist_ok=True)
            wealth_grid_for_vfi = np.geomspace(1e-6, 1000001, num_wealth_points)
            model_params = {
                'interest_rate': interest_rate, 'sigma': sigma,
                'wealth_grid': wealth_grid_for_vfi, 
                'borrowing_limit': 0
            }
            profile_params_list = []
            global agent_profiles
            for name in unique_profiles_to_compute:
                params = agent_profiles[name].copy()
                params['name'] = name
                log_filename = f"R_{interest_rate}_{name}_vfi_log.txt"
                params['log_path'] = os.path.join(vfi_log_dir, log_filename)
                profile_params_list.append(params)

            results = Parallel(n_jobs=-1, verbose=51)(delayed(calculate_vfi_for_profile)(prof_params, model_params) for prof_params in profile_params_list)
            for profile_name, V, g in results:
                self.vfi_cache[profile_name] = (V, g)

            end_time = time.time()
            print(f"--- VFI Pre-computation finished in {end_time - start_time:.2f} seconds. Cache is populated. ---")

        # --- Model Parameters ---
        self.num_agents = sum(population_composition.values())
        self.interest_rate = interest_rate
        self.sigma = sigma
        self.max_wealth = 1000001
        self.borrowing_limit = 0
        self.wealth_dist = wealth_dist
        self.wealth_grid = np.geomspace(1e-6, self.max_wealth, num_wealth_points)
        self.peer_comparison_active = peer_comparison_active
        self.social_norm_active = social_norm_active
        self.information_diffusion_active = information_diffusion_active
        self.peer_comparison_strength = peer_comparison_strength
        self.social_norm_strength = social_norm_strength
        self.information_diffusion_strength = information_diffusion_strength
        self.vfi_recalculation_interval = vfi_recalculation_interval
        
        # --- Mesa Components ---
        self.setup_network(network, network_params)
        self.schedule = RandomActivation(self)
        
        # --- Create Agents ---
        self.create_agents(population_composition)
        
        # --- Data Collection ---
        def get_consumption(agent):
            if agent.previous_wealth is None: return 0
            return max(agent.model.interest_rate * agent.previous_wealth - agent.savings, 1e-9)
        def get_utility(agent):
            # Handle cases where consumption might lead to invalid utility (e.g., log(0))
            utility_val = agent.utility(get_consumption(agent))
            if not np.isfinite(utility_val):
                return np.nan # Return NaN or 0 if utility is invalid
            return utility_val

        self.datacollector = DataCollector(
             model_reporters={
                "Average Wealth": lambda m: np.mean([agent.wealth for agent in m.schedule.agents]),
                "Median Wealth": lambda m: np.median([agent.wealth for agent in m.schedule.agents]),
                "Std. Dev. Wealth": lambda m: np.std([agent.wealth for agent in m.schedule.agents]),
                "Average Consumption": lambda m: np.mean([get_consumption(agent) for agent in m.schedule.agents]),
                "Average Utility": lambda m: np.mean([get_utility(agent) for agent in m.schedule.agents]),
                "Average Savings": lambda m: np.mean([agent.savings for agent in m.schedule.agents]),
                "Gini_Coefficient": compute_gini,
                "Wealth_Quantile_10": lambda m: get_wealth_quantile(m, 0.10),
                "Wealth_Quantile_90": lambda m: get_wealth_quantile(m, 0.90),
            },
            agent_reporters={
                "Wealth": "wealth", 
                "Savings": "savings", 
                "Policy_Savings": "policy_savings",
                "Socially_Adjusted_Goal": "socially_adjusted_savings_goal",
                "Consumption": get_consumption,
                "Total Resources": lambda a: a.previous_wealth * a.model.interest_rate if a.previous_wealth is not None else 0,
                "R_star": "R_star", 
                "Interest_Rate": lambda a: a.model.interest_rate,
                "Utility": get_utility, 
                "Previous_Wealth": "previous_wealth", 
                "Profile": "profile_name",
                "Original_Profile": "original_profile",
                "Beta": "beta",
                "Financial_Literacy": "financial_literacy"
            }
        )
        print("Model initialized")

    def setup_network(self, network_type, params):
        """Initializes the social network for the agents."""
        print(f"Setting up '{network_type}' network...")
        if network_type == 'grid':
            # Create a spatial grid that is large enough to hold all agents
            grid_size = math.ceil(math.sqrt(self.num_agents))
            self.grid = MultiGrid(grid_size, grid_size, torus=True)
            print(f"  > Created {grid_size}x{grid_size} MultiGrid.")
        
        elif network_type == 'watts_strogatz':
            # Use default parameters if none are provided
            if params is None:
                params = {'k': 4, 'p': 0.1}
            
            # Create a Watts-Strogatz graph
            # The nodes of the graph are integers from 0 to N-1
            g = nx.watts_strogatz_graph(n=self.num_agents, 
                                       k=params.get('k', 4), 
                                       p=params.get('p', 0.1),
                                       seed=self._seed)
            self.grid = NetworkGrid(g)
            print(f"  > Created Watts-Strogatz network with n={self.num_agents}, k={params.get('k', 4)}, p={params.get('p', 0.1)}.")
            
        else:
            raise ValueError(f"Unknown network type: {network_type}")

    def create_agents(self, population_composition):
        """
        Creates a population of SavingAgents, adds them to the schedule,
        and places them in the network.
        """
        print("Creating agent population...")
        global agent_profiles

        agent_id_counter = 0
        for profile_name, count in population_composition.items():
            if profile_name not in agent_profiles:
                print(f"Warning: Profile '{profile_name}' not found. Skipping.")
                continue

            print(f"  Creating {count} agent(s) with profile: '{profile_name}'")
            profile_params = agent_profiles[profile_name]

            for i in range(count):
                # Determine initial wealth
                rand_num = self.random.random()
                cumulative_prob = 0
                init_wealth = 0
                for prob, wealth_range in self.wealth_dist:
                    cumulative_prob += prob
                    if rand_num <= cumulative_prob:
                        init_wealth = self.random.randint(wealth_range[0], wealth_range[1])
                        break
                else:
                    init_wealth = self.random.randint(self.wealth_dist[-1][1][0], self.wealth_dist[-1][1][1])

                agent_vfi_iterations = profile_params.get("vfi_iterations", 50)

                agent = SavingAgent(
                    unique_id=agent_id_counter,
                    model=self,
                    profile_name=profile_name,
                    beta=profile_params["beta"],
                    delta=profile_params["delta"],
                    init_wealth=init_wealth,
                    sigma=self.sigma,
                    borrowing_limit=self.borrowing_limit,
                    financial_literacy=profile_params["financial_literacy"],
                    iterations=agent_vfi_iterations
                )
                self.schedule.add(agent)

                if isinstance(self.grid, MultiGrid):
                    # Get a list of all empty cells
                    empty_cells = self.grid.empties
                    # If there are empty cells, pick one at random
                    if empty_cells:
                        pos = self.random.choice(list(empty_cells))
                        # Place the agent there
                        self.grid.place_agent(agent, pos)
                elif isinstance(self.grid, NetworkGrid):
                    # Place the agent on the network node corresponding to its unique_id
                    self.grid.place_agent(agent, agent.unique_id)
                # --------------------------------

                agent_id_counter += 1

        print(f"Total agents created and placed in network: {len(self.schedule.agents)}")


    def _recalculate_vfi_for_changed_agents(self):
        """
        Identifies agents whose preferences (beta) have changed, calculates new VFI
        solutions for the new unique preference profiles, and updates the cache.
        """
        print(f"\n--- Checking for VFI recalculation at step {self.schedule.steps} ---")
        
        # Use a set to find unique new profiles that need to be computed
        profiles_to_compute = set()
        
        for agent in self.schedule.agents:
            # Create a new, unique profile name based on the agent's current preferences
            new_profile_name = f"dynamic_beta_{agent.beta:.4f}_delta_{agent.delta:.4f}"
            
            if new_profile_name not in self.vfi_cache:
                profiles_to_compute.add((new_profile_name, agent.beta, agent.delta))
            
            agent.profile_name = new_profile_name

        if not profiles_to_compute:
            print("No new agent profiles to compute. All preferences are covered by the cache.")
            return

        print(f"Found {len(profiles_to_compute)} new unique agent profiles to calculate.")
        
        # --- Prepare and run the parallel VFI ---
        vfi_log_dir = os.path.join(output_dir_text, "vfi_logs")
        model_params = {
            'interest_rate': self.interest_rate, 'sigma': self.sigma,
            'wealth_grid': self.wealth_grid, 'borrowing_limit': self.borrowing_limit
        }
        
        profile_params_list = []
        global agent_profiles
        for name, beta, delta in profiles_to_compute:
            
            agent_profiles[name] = {"beta": beta, "delta": delta, "vfi_iterations": 1000, "financial_literacy": 0} 
            
            params = agent_profiles[name].copy()
            params['name'] = name
            log_filename = f"R_{self.interest_rate}_{name}_step_{self.schedule.steps}_vfi_log.txt"
            params['log_path'] = os.path.join(vfi_log_dir, log_filename)
            profile_params_list.append(params)

        start_time = time.time()
        results = Parallel(n_jobs=8, verbose=51)(
            delayed(calculate_vfi_for_profile)(prof_params, model_params) for prof_params in profile_params_list
        )

        for profile_name, V, g in results:
            self.vfi_cache[profile_name] = (V, g)
        
        end_time = time.time()
        print(f"--- On-the-fly VFI finished in {end_time - start_time:.2f} seconds. Cache updated. ---")
        

    def step(self):
        """
        Advances the model by one time step.
        """
        if self.social_norm_active and self.schedule.steps > 0 and \
           self.schedule.steps % self.vfi_recalculation_interval == 0:
            self._recalculate_vfi_for_changed_agents()

        print("Model step start", flush=True)  # Debug print
        self.datacollector.collect(self)    # Collect data
        self.schedule.step()    # Advance the agent (and scheduler)
        print("Model step end", flush=True)  # Debug print


# ----------------------------------------------------------
# --- Simulation and Data Export Block ---
# ----------------------------------------------------------

# --- 1. Define Experimental Parameters ---

# Define Agent Profiles
agent_profiles = {
    "planner": {"beta": 0.97, "delta": 0.96, "vfi_iterations": 1000, "financial_literacy": 0.9},
    "moderate": {"beta": 0.90, "delta": 0.91, "vfi_iterations": 1000, "financial_literacy": 0.6},
    "procrastinator": {"beta": 0.78, "delta": 0.95, "vfi_iterations": 1000, "financial_literacy": 0.3},
    "inverse procrastinator": {"beta": 0.96, "delta": 0.85, "vfi_iterations": 1000, "financial_literacy": 0.5},
    "impulsive": {"beta": 0.60, "delta": 0.80, "vfi_iterations": 1000, "financial_literacy": 0.1},
}

# Define Economic Conditions
sigma = 1
wealth_dist = [
    (0.4152, (0, 9999)),
    (0.4772, (10000, 99999)),
    (0.1031, (100000, 999999)),
    (0.0045, (1000000, 1000001))
]

# Define the population for the experiment
population_to_simulate = {
    "planner": 15,
    "moderate": 35, 
    "procrastinator": 30, 
    "inverse procrastinator": 10,
    "impulsive": 10
}

# Define the conditions to iterate over
interest_rates_to_test = [1.05, 1.12] 
SIMULATION_STEPS = 200
NUM_WEALTH_POINTS = 1000

experiments = {
    "baseline": {
        "peer_comparison_active": False, "social_norm_active": False, "information_diffusion_active": False
    },
    "peer_comparison_only": {
        "peer_comparison_active": True, "social_norm_active": False, "information_diffusion_active": False
    },
    "social_norms_only": {
        "peer_comparison_active": False, "social_norm_active": True, "information_diffusion_active": False
    },
    "info_diffusion_only": {
        "peer_comparison_active": False, "social_norm_active": False, "information_diffusion_active": True
    },
    "all_interactions": {
        "peer_comparison_active": True, "social_norm_active": True, "information_diffusion_active": True
    }
}

experiments_to_run = ["baseline"]

# --- 2. Setup Output Directories ---
output_dir_csv = "output_csv"
os.makedirs(output_dir_csv, exist_ok=True)
output_dir_text = "output_text"
os.makedirs(output_dir_text, exist_ok=True)
log_filepath = os.path.join(output_dir_text, "simulation_run_log.txt")

print(f"Starting simulation. All detailed output will be saved to: {log_filepath}")

# --- 3. Run Simulation with Logging ---

with open(log_filepath, "w") as log_file:
    # --- REDIRECT STDOUT TO THE LOG FILE ---
    original_stdout = sys.stdout
    sys.stdout = log_file

    print("="*50)
    print("SIMULATION RUN STARTED...")
    print(f"Current Time: {time.ctime()}")
    print("="*50 + "\n")

    NUM_REPLICATIONS = 30
    seeds = range(1, NUM_REPLICATIONS + 1)

    total_runs = len(interest_rates_to_test) * len(experiments_to_run) * NUM_REPLICATIONS
    completed_runs = 0
    start_time = time.time()

    # Loop through each experimental condition
    for rate in interest_rates_to_test:

        print(f"\\n{'='*25} PRE-COMPUTING VFI FOR R = {rate} {'='*25}")
        temp_model = SavingModel(
            population_composition=population_to_simulate,
            interest_rate=rate,
            sigma=sigma,
            wealth_dist=wealth_dist,
            num_wealth_points=NUM_WEALTH_POINTS,
            network='watts_strogatz',  # Specify the network type
            network_params={'k': 4, 'p': 0.1}, # Define the network's parameters
            seed=1
        )

        precomputed_cache = temp_model.vfi_cache
        print("--- VFI Cache Generation Complete ---")

        print("--- Exporting V and g functions from cache... ---")
        output_dir_v_g = os.path.join(output_dir_csv, "v_g_functions")
        os.makedirs(output_dir_v_g, exist_ok=True)
        for profile_name, (V, g) in precomputed_cache.items():
            v_g_base_filename = f"R_{rate}_{profile_name}"
            v_func_path = os.path.join(output_dir_v_g, f"{v_g_base_filename}_value_function.npy")
            g_func_path = os.path.join(output_dir_v_g, f"{v_g_base_filename}_policy_function.npy")
            np.save(v_func_path, V)
            np.save(g_func_path, g)
            print(f"  > Saved V and g for profile '{profile_name}'")


        

        for run_id, seed in enumerate(seeds):

            print(f"\n{'#'*25} STARTING REPLICATION {run_id + 1}/{NUM_REPLICATIONS} (Seed: {seed}) {'#'*25}")

            for run_name, settings in experiments.items():
                if run_name not in experiments_to_run:
                    print(f"\n--- SKIPPING EXPERIMENT: '{run_name}' ---")
                    continue
                print(f"\n{'='*20} RUNNING EXPERIMENT: '{run_name}' | Interest Rate (R) = {rate} {'='*20}")
            
                model = SavingModel(
                    population_composition=population_to_simulate,
                    interest_rate=rate,
                    sigma=sigma,
                    wealth_dist=wealth_dist,
                    num_wealth_points=NUM_WEALTH_POINTS,
                    network='watts_strogatz', 
                    network_params={'k': 4, 'p': 0.1},
                    seed=seed,
                    peer_comparison_active=settings["peer_comparison_active"],
                    social_norm_active=settings["social_norm_active"],
                    information_diffusion_active=settings["information_diffusion_active"],
                    social_norm_strength=0.15,
                    peer_comparison_strength=0.15,
                    information_diffusion_strength=0.15,
                    vfi_recalculation_interval=10,
                    vfi_cache=precomputed_cache
                )   
                
                # Run the model for the specified number of steps
                for i in range(SIMULATION_STEPS):
                    print(f"\n--- MODEL STEP {i} ---")
                    model.step()
                    
                print(f"\n--- Simulation Complete for R={rate}. Exporting data... ---")

                # Retrieve and save data
                agent_data = model.datacollector.get_agent_vars_dataframe()
                model_data = model.datacollector.get_model_vars_dataframe()
                
                base_filename = f"run_{run_name}_R_{rate}_rep_{run_id + 1}_pop_{len(model.schedule.agents)}agents_steps_{SIMULATION_STEPS}"
                agent_data_filepath = os.path.join(output_dir_csv, f"{base_filename}_agent_data.csv")
                model_data_filepath = os.path.join(output_dir_csv, f"{base_filename}_model_data.csv")
                
                agent_data.to_csv(agent_data_filepath)
                model_data.to_csv(model_data_filepath)
                
                print(f"Successfully saved Agent Data to: {agent_data_filepath}")
                print(f"Successfully saved Model Data to: {model_data_filepath}")

                completed_runs += 1
                progress_percent = (completed_runs / total_runs) * 100
                elapsed_seconds = time.time() - start_time
                elapsed_h = int(elapsed_seconds // 3600)
                elapsed_m = int((elapsed_seconds % 3600) // 60)
                elapsed_s = int(elapsed_seconds % 60)
                time_str = f"{elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:02d}"
                print(f"Overall Progress: [{completed_runs}/{total_runs}] {progress_percent:.1f}% Completed in {time_str}", end='\\r', file=sys.stderr)
                
                if run_name == "baseline": # Only save V/g on the first run to avoid redundancy
                    print("--- Exporting V and g functions from cache... ---")
                    output_dir_v_g = os.path.join(output_dir_csv, "v_g_functions")
                    os.makedirs(output_dir_v_g, exist_ok=True)
                    for profile_name, (V, g) in model.vfi_cache.items():
                        v_g_base_filename = f"R_{rate}_{profile_name}"
                        v_func_path = os.path.join(output_dir_v_g, f"{v_g_base_filename}_value_function.npy")
                        g_func_path = os.path.join(output_dir_v_g, f"{v_g_base_filename}_policy_function.npy")
                        np.save(v_func_path, V)
                        np.save(g_func_path, g)
                        print(f"  > Saved V and g for profile '{profile_name}'")

    print("\n" + "="*50)
    print("ALL SIMULATIONS COMPLETE. ALL DATA EXPORTED.")
    print(f"Output files are in the '{output_dir_csv}' directory.")
    print("="*50 + "\n")

# --- RESTORE STDOUT TO THE CONSOLE ---
sys.stdout = original_stdout

print("Process finished successfully.")
print(f"All data saved to '{output_dir_csv}'.")
print(f"Full simulation log saved to '{log_filepath}'.")