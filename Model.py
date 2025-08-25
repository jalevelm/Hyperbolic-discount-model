import sys 
import os 
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
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
    def __init__(self, unique_id, model, profile_name, beta, delta, init_wealth, sigma, borrowing_limit, iterations=50):
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
        super().__init__(unique_id, model)  # Call the superclass constructor

        self.profile_name = profile_name
        # --- Agent Preferences and State ---
        self.beta = beta        # Present bias
        self.delta = delta      # Discount factor
        self.sigma = sigma      # Inverse of elasticity of intertemporal substitution

        # --- Initial Conditions ---
        self.wealth = init_wealth   # Current wealth
        self.init_wealth = init_wealth # Initial wealth (stored for reference)
        self.borrowing_limit = borrowing_limit  # Borrowing constraint
        self.savings = 0        # Initial savings (updated each step)
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
            # --- THIS LINE IS CORRECTED ---
            # For a grid, neighbors are in adjacent cells. moore=True includes diagonals.
            return self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        elif isinstance(self.model.grid, NetworkGrid):
            # For a network, neighbors are agents connected by an edge.
            # The grid returns their unique_id, so we get the agent objects
            # from the model's scheduler.
            neighbor_ids = self.model.grid.get_neighbors(self.unique_id, include_center=False)
            return [self.model.schedule.agents[i] for i in neighbor_ids]
        return []

    def step(self):
        """
        Advances the agent by one time step, printing detailed logs of its actions.
        """
        # --- Print Initial State ---
        if self.step_count == 0:
            print(f"Agent {self.unique_id} ({self.profile_name}): First Step - Beta: {self.beta}, Delta: {self.delta}, R_star: {self.R_star}")
            neighbors = self.get_neighbors()
            print(f"Agent {self.unique_id}: Has {len(neighbors)} neighbors.")
        
        print(f"Agent {self.unique_id} ({self.profile_name}): Step start. Wealth: {self.wealth:.4f}, Previous Savings: {self.previous_savings}")
        print(f"Agent {self.unique_id}: R = {self.model.interest_rate}, R_star = {self.R_star}")

        # --- Retrieve VFI Solution (First Step Only) ---
        if not self.value_function_calculated:
            if self.profile_name in self.model.vfi_cache:
                self.V, self.g = self.model.vfi_cache[self.profile_name]
                self.value_function_calculated = True
                print(f"Agent {self.unique_id} ({self.profile_name}): Cache HIT. Retrieved V and g functions.")
            else:
                raise Exception(f"CRITICAL ERROR: VFI for profile '{self.profile_name}' not found in cache for Agent {self.unique_id}.")

        # --- Agent's Decision (Every Step) ---
        wealth_grid = self.model.wealth_grid
        wealth_index = np.argmin(np.abs(wealth_grid - self.wealth))
        optimal_savings = self.g[wealth_index]
        self.savings = np.clip(optimal_savings, self.borrowing_limit, self.model.interest_rate * self.wealth)
        
        wealth_with_interest = self.model.interest_rate * self.wealth
        consumption = max(wealth_with_interest - self.savings, 1e-9)
        percentage_consumed = (consumption / (wealth_with_interest + 1e-9)) * 100
        percentage_saved = (self.savings / (wealth_with_interest + 1e-9)) * 100

        # --- Print Detailed Decision Logs ---
        print(f"Agent {self.unique_id}: Step {self.step_count}")
        print(f"  Previous Wealth: {self.previous_wealth:.4f}")
        print(f"  Optimal Savings (from policy): {optimal_savings:.4f}")
        print(f"  Calculated Savings (after clipping): {self.savings:.4f}")
        print(f"  Consumption: {consumption:.4f}")
        
        # --- State Updates ---
        self.previous_wealth = self.wealth
        self.wealth = self.savings
        self.previous_savings = self.savings

        # --- Print Final Summary ---
        print(f"Agent {self.unique_id}: Step end. | Total resources: {wealth_with_interest:.2f}, Consumption: {consumption:.2f}, Savings: {self.savings:.2f}, Wealth end of step: {self.wealth:.2f}")
        print(f"Percentage consumed: {percentage_consumed:.2f}%, percentage saved: {percentage_saved:.2f}%\n")

        # --- History Tracking ---
        self.consumption_history.append(consumption)
        self.wealth_history.append(self.wealth)
        self.utility_history.append(self.utility(consumption))
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

# --- Standalone VFI Function (now with its own logging) ---
def calculate_vfi_for_profile(profile_params, model_params):
    """
    Performs VFI and writes its detailed progress to a unique log file.
    """
    # Unpack parameters, including the new log_path
    log_path = profile_params['log_path']
    profile_name = profile_params['name']
    
    # This try...finally block ensures that standard output is restored
    # for the worker process, even if an error occurs.
    original_stdout = sys.stdout
    try:
        # Redirect all print statements within this block to the unique log file
        with open(log_path, 'w') as log_file:
            sys.stdout = log_file

            # --- Start of Original VFI Logic ---
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

            # (Nested utility and optimize_savings functions are the same as before)
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
            # --- End of Original VFI Logic ---
    finally:
        # Restore standard output for the worker process
        sys.stdout = original_stdout

class SavingModel(Model):
    """
    A Mesa model simulating the saving behavior of agents with
    time-inconsistent preferences, based on the Cao-Werning (2018) model.

    The model features a single agent (for now, for debugging purposes) that makes
    consumption and savings decisions in discrete time over a finite horizon.
    The agent has hyperbolic discounting preferences and faces a constant
    interest rate and a borrowing constraint.

    Attributes:
        num_agents (int): The number of agents in the model (currently fixed at 1).
        grid (MultiGrid): A Mesa MultiGrid object (not actively used in this
            single-agent version, but retained for potential future extensions).
        schedule (RandomActivation): A scheduler that activates agents in random
            order (in this case, only one agent).
        interest_rate (float): The constant gross interest rate for saving and
            borrowing.
        sigma (float): the inverse of the elasticity of intertemporal substitution for the
            agent's utility function.
        max_wealth (float): The maximum wealth level allowed in the model.
            Used to define the wealth grid.
        borrowing_limit (float): The minimum allowed wealth level (set to 0).
        wealth_dist (list): A list of tuples defining the distribution of
            initial wealth among agents.  Each tuple contains a probability
            and a wealth range (min, max).
        wealth_grid (np.ndarray): A discretized grid of possible wealth levels,
            used for value function iteration and interpolation.
        datacollector (DataCollector): A Mesa DataCollector object for collecting
            model and agent-level data during the simulation.
    """
    
    def __init__(self, population_composition, interest_rate, sigma, wealth_dist, 
             num_wealth_points=100, network='grid', network_params=None):
        super().__init__()

        # --- Model Parameters ---
        self.num_agents = sum(population_composition.values())
        self.interest_rate = interest_rate
        self.sigma = sigma
        self.max_wealth = 1000001
        self.borrowing_limit = 0
        self.wealth_dist = wealth_dist
        self.wealth_grid = np.geomspace(1e-6, self.max_wealth, num_wealth_points)
        
        # --- Mesa Components ---
        # --- NEW: Call network setup method ---
        self.setup_network(network, network_params)
        self.schedule = RandomActivation(self)
        
        print("--- Starting VFI Pre-computation for all profiles ---")
        start_time = time.time()
        
        # --- Parallel VFI Pre-computation ---
        self.vfi_cache = {}
        unique_profiles_to_compute = population_composition.keys()

        vfi_log_dir = os.path.join(output_dir_text, "vfi_logs")
        os.makedirs(vfi_log_dir, exist_ok=True)
        
        # Prepare arguments for the parallel function
        model_params = {
            'interest_rate': self.interest_rate, 'sigma': self.sigma,
            'wealth_grid': self.wealth_grid, 'borrowing_limit': self.borrowing_limit
        }
        
        profile_params_list = []
        global agent_profiles
        for name in unique_profiles_to_compute:
            params = agent_profiles[name].copy()
            params['name'] = name
            log_filename = f"R_{interest_rate}_{name}_vfi_log.txt"
            params['log_path'] = os.path.join(vfi_log_dir, log_filename)
            profile_params_list.append(params)
            

        # Run the VFI for all profiles in parallel [cite: 32]
        # Each call to calculate_vfi_for_profile runs on a separate core
        results = Parallel(n_jobs=-1, verbose=51)(
            delayed(calculate_vfi_for_profile)(prof_params, model_params) for prof_params in profile_params_list
        )

        # Pre-populate the cache with the results [cite: 37-38]
        for profile_name, V, g in results:
            self.vfi_cache[profile_name] = (V, g)
        
        end_time = time.time()
        print(f"--- VFI Pre-computation finished in {end_time - start_time:.2f} seconds. Cache is populated. ---")

        # --- Create Agents (who will now all get cache hits) ---
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
                "Consumption": get_consumption,
                "Total Resources": lambda a: a.previous_wealth * a.model.interest_rate if a.previous_wealth is not None else 0,
                "R_star": "R_star", 
                "Interest_Rate": lambda a: a.model.interest_rate,
                "Utility": get_utility, 
                "Previous_Wealth": "previous_wealth", 
                "Profile": "profile_name"
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
                                       p=params.get('p', 0.1))
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

        # --- NEW: Explicitly manage agent IDs for network mapping ---
        agent_id_counter = 0
        for profile_name, count in population_composition.items():
            if profile_name not in agent_profiles:
                print(f"Warning: Profile '{profile_name}' not found. Skipping.")
                continue

            print(f"  Creating {count} agent(s) with profile: '{profile_name}'")
            profile_params = agent_profiles[profile_name]

            for i in range(count):
                # Determine initial wealth
                rand_num = random.random()
                cumulative_prob = 0
                init_wealth = 0
                for prob, wealth_range in self.wealth_dist:
                    cumulative_prob += prob
                    if rand_num <= cumulative_prob:
                        init_wealth = random.randint(wealth_range[0], wealth_range[1])
                        break
                else:
                    # Fallback for floating point precision issues
                    init_wealth = random.randint(self.wealth_dist[-1][1][0], self.wealth_dist[-1][1][1])

                agent_vfi_iterations = profile_params.get("vfi_iterations", 50)

                # --- MODIFIED: The unique_id is now our counter ---
                agent = SavingAgent(
                    unique_id=agent_id_counter, # Use the counter for the ID
                    model=self,
                    profile_name=profile_name,
                    beta=profile_params["beta"],
                    delta=profile_params["delta"],
                    init_wealth=init_wealth,
                    sigma=self.sigma,
                    borrowing_limit=self.borrowing_limit,
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
                # --------------------------------

                agent_id_counter += 1

        print(f"Total agents created and placed in network: {len(self.schedule.agents)}")
        


    def step(self):
        """
        Advances the model by one time step.

        This involves:
        1. Collecting data from the current state.
        2. Advancing the agent (which performs its saving/consumption decision).
        """
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
    "planner": {"beta": 0.97, "delta": 0.96, "vfi_iterations": 100},
    "moderate": {"beta": 0.90, "delta": 0.91, "vfi_iterations": 100},
    "procrastinator": {"beta": 0.78, "delta": 0.95, "vfi_iterations": 100},
    "inverse procrastinator": {"beta": 0.96, "delta": 0.85, "vfi_iterations": 100},
    "impulsive": {"beta": 0.60, "delta": 0.80, "vfi_iterations": 100},
}

# Define Economic Conditions
sigma = 0.4387
wealth_dist = [
    (0.4152, (0, 9999)),
    (0.4772, (10000, 99999)),
    (0.1031, (100000, 999999)),
    (0.0045, (1000000, 1000001))
]

# Define the population for the experiment
population_to_simulate = {
    "planner": 20,
    "moderate": 30,
    "procrastinator": 20,
    "inverse procrastinator": 15,
    "impulsive": 15
}

# Define the conditions to iterate over
# Note: For testing, you might want to use just one rate, e.g., [1.05]
interest_rates_to_test = [1.10]
SIMULATION_STEPS = 12

# --- 2. Setup Output Directories ---
output_dir_csv = "output_csv"
os.makedirs(output_dir_csv, exist_ok=True)
output_dir_text = "output_text"
os.makedirs(output_dir_text, exist_ok=True)
log_filepath = os.path.join(output_dir_text, "simulation_run_log.txt")

# This print statement will appear in your console
print(f"Starting simulation. All detailed output will be saved to: {log_filepath}")

# --- 3. Run Simulation with Logging ---

# This 'with' block handles the log file redirection
with open(log_filepath, "w") as log_file:
    # --- REDIRECT STDOUT (all print statements) TO THE LOG FILE ---
    original_stdout = sys.stdout
    sys.stdout = log_file

    print("="*50)
    print("SIMULATION RUN STARTED...")
    print(f"Current Time: {time.ctime()}")
    print("="*50 + "\n")

    # Loop through each experimental condition
    for rate in interest_rates_to_test:
        print(f"\n--- Running Simulation for Interest Rate (R) = {rate} ---")
        
        # Create a fresh model instance. This will trigger the parallel VFI pre-computation.
        model = SavingModel(
            population_composition=population_to_simulate,
            interest_rate=rate,
            sigma=sigma,
            wealth_dist=wealth_dist,
            num_wealth_points = 300 
        )   
        
        # Run the model for the specified number of steps
        for i in range(SIMULATION_STEPS):
            print(f"\n--- MODEL STEP {i} ---")
            model.step()
            
        print(f"\n--- Simulation Complete for R={rate}. Exporting data... ---")

        # Retrieve and save data
        agent_data = model.datacollector.get_agent_vars_dataframe()
        model_data = model.datacollector.get_model_vars_dataframe()
        
        base_filename = f"R_{rate}_pop_{len(model.schedule.agents)}agents_steps_{SIMULATION_STEPS}"
        agent_data_filepath = os.path.join(output_dir_csv, f"{base_filename}_agent_data.csv")
        model_data_filepath = os.path.join(output_dir_csv, f"{base_filename}_model_data.csv")
        
        agent_data.to_csv(agent_data_filepath)
        model_data.to_csv(model_data_filepath)
        
        print(f"Successfully saved Agent Data to: {agent_data_filepath}")
        print(f"Successfully saved Model Data to: {model_data_filepath}")
        
        print("--- Exporting V and g functions from cache... ---")
        output_dir_v_g = os.path.join(output_dir_csv, "v_g_functions")
        os.makedirs(output_dir_v_g, exist_ok=True)

        # Loop through the pre-computed items in the cache
        for profile_name, (V, g) in model.vfi_cache.items():
            # Create descriptive filenames for each profile
            v_g_base_filename = f"R_{rate}_{profile_name}"
            v_func_path = os.path.join(output_dir_v_g, f"{v_g_base_filename}_value_function.npy")
            g_func_path = os.path.join(output_dir_v_g, f"{v_g_base_filename}_policy_function.npy")
            
            # Save the V and g arrays to .npy files
            np.save(v_func_path, V)
            np.save(g_func_path, g)
            print(f"  > Saved V and g for profile '{profile_name}'")

    print("\n" + "="*50)
    print("ALL SIMULATIONS COMPLETE. ALL DATA EXPORTED.")
    print(f"Output files are in the '{output_dir_csv}' directory.")
    print("="*50 + "\n")

# --- RESTORE STDOUT TO THE CONSOLE ---
sys.stdout = original_stdout

# These final print statements will appear in your console
print("Process finished successfully.")
print(f"All data saved to '{output_dir_csv}'.")
print(f"Full simulation log saved to '{log_filepath}'.")