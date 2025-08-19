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

    def step(self):
        """
        Advances the agent by one time step. This encompasses:
        1. Calculating the value and policy functions (if not already done).
        2. Determining optimal savings based on the policy function.
        3. Updating the agent's wealth based on its savings decision.
        4. Updating internal tracking variables.

        The agent's decision-making process aims to approximate a Markov
        equilibrium in a dynamic savings game with hyperbolic discounting.
        """

        # --- Initialization (First Step Only) ---
        if self.step_count == 0:    
            print(f"Agent {self.unique_id} ({self.profile_name}): First Step - Beta: {self.beta}, Delta: {self.delta}, R_star: {self.R_star}", flush=True)
        
        
        print(f"Agent {self.unique_id} ({self.profile_name}): Step start. Wealth: {self.wealth}, Previous Savings: {self.previous_savings}", flush=True)
        print(f"Agent {self.unique_id}: R = {self.model.interest_rate}, R_star = {self.R_star}", flush=True)

        wealth_grid = self.model.wealth_grid    # Access the pre-defined wealth grid

        # --- Value Function and Policy Function Calculation (One-Time) ---
        if not self.value_function_calculated:
            #print(f"Agent {self.unique_id}: Starting value iteration...")
            # Initialize value and policy functions as NumPy arrays
            V = np.zeros_like(wealth_grid)
            g = np.zeros_like(wealth_grid)
         
            iterations_to_snapshot = [1, 50 ,100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 750, 1000, 1250, 1500]
    

            # --- Value Iteration Algorithm ---
            tolerance = 1e-6  # Convergence tolerance
            iteration_count = 0   # Iteration counter

            total_optimization_time = 0  # Accumulate optimization times
            total_inner_loop_time = 0

            outer_loop_start_time = time.time()

            print(f"Agent {self.unique_id}: Starting value iteration with wealth_grid size {len(wealth_grid)} and max_iters {self.max_vfi_iterations}", flush=True)

            for _ in range(self.max_vfi_iterations):
                iteration_count += 1 
                V_old = V.copy()    # Store the previous iteration's value function
                g_old_this_iter = g.copy()

                inner_loop_start_time = time.time()

                # Iterate over all possible wealth levels in the grid
                for i, k_val in enumerate(wealth_grid):

                    if i % (len(wealth_grid) // 10) == 0 and iteration_count == 1 : # Ensure it's iteration_count == 1
                        print(f"Agent {self.unique_id}: VFI Iter {iteration_count}, Processing wealth_grid index {i}/{len(wealth_grid)}", flush=True)
        
                    # Find the optimal savings and continuation value using the
                    # current value function (V) and the agent's parameters.
                    continuation_value, next_k, _ = self.optimize_savings(k_val, V_old, wealth_grid) # NOTE: V here is the V being updated, not V_old

                    # Calculate consumption based on the budget constraint
                    consumption = self.model.interest_rate * k_val - next_k
                    consumption = max(consumption, 1e-9)    # Ensure consumption >= 0

                    # Update the value function: current utility + discounted future utility
                    V[i] = self.utility(consumption) + self.delta * continuation_value                                       #or self.beta
                    g[i] = next_k # Store the optimal next-period wealth (savings)

                inner_loop_end_time = time.time()
                total_inner_loop_time += (inner_loop_end_time - inner_loop_start_time)


               
                if iteration_count in iterations_to_snapshot:
                    self.V_snapshots[iteration_count] = V.copy()
                    self.g_snapshots[iteration_count] = g.copy()


                diff_V = np.max(np.abs(V - V_old)) 
                abs_diff_g_array = np.abs(g - g_old_this_iter)
                diff_g = np.max(abs_diff_g_array)
                if diff_g > 0: # Avoid error if diff_g is zero (no single max index then)
                    idx_max_diff_g = np.argmax(abs_diff_g_array)
                    k_val_max_diff_g = wealth_grid[idx_max_diff_g]
                else:
                    k_val_max_diff_g = np.nan # Or some other placeholder if diff_g is 0


                
                if iteration_count % 20 == 0: 
                    print(f"Agent {self.unique_id}: VFI Iteration {iteration_count}/{self.max_vfi_iterations}, Diff_V: {diff_V:.4e}, Diff_g: {diff_g:.4e} (at k={k_val_max_diff_g:.2f})", flush=True)
                
                # --- Convergence and other checks ---
                if np.any(np.isnan(V)) or np.any(np.isinf(V)):
                    print(f"Agent {self.unique_id}: NaN/Inf detected in V after iteration {iteration_count}! Diff_V: {diff_V:.4e}, Diff_g: {diff_g:.4e} (at k={k_val_max_diff_g:.2f})", flush=True)
                    break

                if np.any(np.abs(V) > 1e6): # Check for divergence
                    print(f"Agent {self.unique_id}: Value function is likely diverging at iteration {iteration_count}! Diff_V: {diff_V:.4e}, Diff_g: {diff_g:.4e} (at k={k_val_max_diff_g:.2f})", flush=True)
                    raise ValueError("Value function is likely diverging")


                if diff_V < tolerance:   # Check for convergence based on Value Function
                    print(f"Agent {self.unique_id}: Value function converged after {iteration_count} iterations. Diff_V: {diff_V:.4e}, Diff_g: {diff_g:.4e} (at k={k_val_max_diff_g:.2f})", flush=True)
                    break    
            else: # Loop finished without break (no convergence)
                # Also ensure k_val_max_diff_g is defined here if the loop runs to completion
                if 'k_val_max_diff_g' not in locals(): # Handle case if loop was very short
                    k_val_max_diff_g = np.nan
                print(f"Agent {self.unique_id}: Value function DID NOT converge after {iteration_count} iterations. Final Diff_V: {diff_V:.4e}, Diff_g: {diff_g:.4e} (at k={k_val_max_diff_g:.2f})", flush=True)

            outer_loop_end_time = time.time()
            total_value_iteration_time = outer_loop_end_time - outer_loop_start_time

            average_optimization_time = total_optimization_time / (iteration_count * len(wealth_grid)) if iteration_count > 0 else 0
            average_inner_loop_time = total_inner_loop_time / iteration_count if iteration_count > 0 else 0


            print(f"Agent {self.unique_id}: Total value iteration took {total_value_iteration_time:.6f} seconds", flush=True)
            print(f"Agent {self.unique_id}: Average optimization time: {average_optimization_time:.8f} seconds", flush=True)
            print(f"Agent {self.unique_id}: Average inner loop time: {average_inner_loop_time:.6f} seconds", flush=True)


            # Store the calculated value and policy functions
            self.V = V
            self.g = g
            self.value_function_calculated = True   # Set flag to avoid recalculation

        # --- Agent's Decision (Every Step) ---

         # Find the index in the wealth grid that is closest to the agent's current wealth
        wealth_index = np.argmin(np.abs(wealth_grid - self.wealth))
        optimal_savings = self.g[wealth_index] # Look up optimal savings in policy function
         # Clip savings to ensure it's within feasible bounds (borrowing limit and max possible wealth)
        self.savings = np.clip(optimal_savings, self.borrowing_limit, self.model.interest_rate * self.wealth)


        # Calculate consumption based on the budget constraint
        wealth_with_interest=self.model.interest_rate * self.wealth
        consumption = self.model.interest_rate * self.wealth - self.savings
        consumption = max(consumption, 1e-9)  # Ensure positive consumption
        percentage_consumed = (consumption / wealth_with_interest) * 100
        percentage_saved = (self.savings / wealth_with_interest) * 100


        # --- Debugging Prints (End of Step) ---
        print(f"Agent {self.unique_id}: Step {self.step_count}", flush=True)
        print(f"  Previous Wealth: {self.previous_wealth:.4f}", flush=True)
        print(f"  Optimal Savings (before clipping): {optimal_savings:.4f}", flush=True)  
        print(f"  Calculated Savings (after clipping): {self.savings:.4f}", flush=True)
        print(f"  New Wealth: {self.wealth:.4f}", flush=True)
        print(f"  Consumption: {consumption:.4f}", flush=True)

        # --- State Updates ---
        self.previous_wealth = self.wealth  # Store current wealth for next step
        self.wealth = self.savings  # Next period's wealth is this period's savings
        self.previous_savings = self.savings    # Store current savings for next step

        

        print(f"Agent {self.unique_id}: Step end. | Total resources available: {wealth_with_interest:.2f}, Consumption: {consumption:.2f}, Savings: {self.savings:.2f},  Wealth at the end of step: {self.wealth:.2f} ", flush=True)
        print(f"Percentage of total resources consumed: {percentage_consumed:.2f}, percentage saved: {percentage_saved:.2f}")

        self.consumption_history.append(consumption)
        self.wealth_history.append(self.wealth) # self.wealth is now end-of-step wealth
        current_step_utility = self.utility(consumption)
        self.utility_history.append(current_step_utility)

        self.step_count += 1    # Increment the step counter

    def utility(self, consumption):
        """
        Calculates the agent's utility from consumption in a single period.

        This uses an isoelastic utility function, a standard form in economics. The specific form
        is:

            u(c) = c^(1-sigma) / (1-sigma)   if sigma != 1
            u(c) = ln(c)                     if sigma == 1

        where:
            c is consumption
            sigma is the inverse of the elasticity of intertemporal substitution.

        Args:
            consumption (float): The level of consumption.

        Returns:
            float: The utility level.

        Raises:
            None (but includes a safeguard against non-positive consumption).
        """
        epsilon = 1e-9  # Small positive value to avoid numerical issues.

        # Ensure consumption is positive (or very close to zero).  The
        # utility function is undefined for zero or negative consumption.
        consumption = max(consumption, epsilon)  

        if self.sigma == 1:
            # Special case: sigma = 1 corresponds to logarithmic utility.
            return np.log(consumption)
        else:
            # General case: isoelastic utility.
            return consumption**(1 - self.sigma) / (1 - self.sigma)

    

    def optimize_savings(self, k, V_old_for_opt, wealth_grid_for_opt): 
        """
        Optimizes the agent's savings decision for a given level of wealth
        by searching over discrete choices for next period's wealth (k_next)
        from the provided wealth_grid.

        This function finds the optimal next-period wealth (k_next) from the grid
        that maximizes the agent's present-biased discounted utility, given its
        current wealth (k) and the value function from the previous iteration (V_old_for_opt).

        Args:
            k (float): The agent's current wealth.
            V_old_for_opt (np.ndarray): The value function from the PREVIOUS VFI iteration.
            wealth_grid_for_opt (np.ndarray): The discretized grid of possible wealth levels.

        Returns:
            tuple: A tuple containing:
                - continuation_value (float): The interpolated value V_old_for_opt(optimal_k_next),
                                             used for the Bellman equation.
                - optimal_k_next (float): The chosen next-period wealth from the grid.
                - final_total_utility (float): The maximized utility (u(c*) + beta*delta*V_old_for_opt(k'*))
                                               achieved with the optimal_k_next.
                                               (Note: the Bellman uses delta*V_old_for_opt(k'*),
                                                the objective function for choice uses beta*delta*V_old_for_opt(k'*))

                Returns (0, k, -np.inf) or similar if no valid option is found.
        """
        beta = self.beta    # Present bias parameter
        delta = self.delta  # Discount factor
        R = self.model.interest_rate   # Gross interest rate
        
        lower_bound_k_next = max(self.borrowing_limit, 0) # Min k_next (savings)
        upper_bound_k_next = R * k # Max k_next (savings can't exceed current resources times interest)

        max_objective_function_val = -np.inf # We are maximizing utility directly here
        
        # Default optimal_k_next: clip current k to feasible bounds.
        # This ensures consumption is non-negative if k is positive.
        # If R*k itself is less than lower_bound_k_next, this will choose lower_bound_k_next.
        optimal_k_next = np.clip(k, lower_bound_k_next, upper_bound_k_next) 

        # Iterate over all possible k_next choices from the grid
        for k_next_candidate in wealth_grid_for_opt:
            # Ensure the candidate for k_next is within feasible saving limits
            if k_next_candidate < lower_bound_k_next or k_next_candidate > upper_bound_k_next:
                continue

            consumption = R * k - k_next_candidate
            
            if consumption <= 1e-9: # Penalize non-positive consumption
                current_objective_function_val = -np.inf 
            else:
                # Value of V_old_for_opt at k_next_candidate for the objective function
                # Note: k_next_candidate is already a grid point, but if wealth_grid_for_opt
                # is not perfectly aligned or if k_next_candidate was somehow not from the grid,
                # interpolation is safer. Since it *is* from the grid, direct indexing
                # could be faster if you find the index, but interp is robust.
                # For future_utility_component, it's beta * (delta * V_old(k_next_candidate))
                # as this is what the agent maximizes.
                val_at_k_next_candidate = np.interp(k_next_candidate, wealth_grid_for_opt, V_old_for_opt)
                future_utility_for_objective = delta * val_at_k_next_candidate

                current_objective_function_val = self.utility(consumption) + beta * future_utility_for_objective
            
            if current_objective_function_val > max_objective_function_val:
                max_objective_function_val = current_objective_function_val
                optimal_k_next = k_next_candidate

        # If no valid choice was found (e.g., all consumptions were non-positive),
        # max_objective_function_val might still be -np.inf.
        # In this case, optimal_k_next might still be its default.
        # We need to ensure optimal_k_next leads to non-negative consumption for calculating final utility.
        if max_objective_function_val == -np.inf:
            # Fallback: save as much as possible up to R*k, or hit borrowing limit, ensuring c>=0
            optimal_k_next = np.clip(R * k, lower_bound_k_next, upper_bound_k_next) 
            # Re-calculate max_objective_function_val for this fallback optimal_k_next
            final_consumption_check = R * k - optimal_k_next
            final_consumption_check = max(final_consumption_check, 1e-9) # Ensure positive for utility calc

            val_at_optimal_k_next_fallback = np.interp(optimal_k_next, wealth_grid_for_opt, V_old_for_opt)
            future_utility_for_objective_fallback = delta * val_at_optimal_k_next_fallback
            max_objective_function_val = self.utility(final_consumption_check) + beta * future_utility_for_objective_fallback


        # Continuation value for the Bellman equation is V_old_for_opt(optimal_k_next)
        continuation_value_for_bellman = np.interp(optimal_k_next, wealth_grid_for_opt, V_old_for_opt)
        
        # The value returned as "final_total_utility" should be the maximized objective value
        final_total_utility_debug = max_objective_function_val

        return continuation_value_for_bellman, optimal_k_next, final_total_utility_debug


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
    
    def __init__(self, population_composition, interest_rate, sigma, wealth_dist, num_wealth_points=100):
        """
        Initializes the SavingModel.

        Args:
            agent_profile (dict): A dictionary containing the parameters
                ('beta' and 'delta') for the agent's hyperbolic discounting
                preferences.
            interest_rate (float): The constant gross interest rate.
            sigma (float): The coefficient of relative risk aversion.
            wealth_dist (list): The distribution of initial wealth.
            num_wealth_points (int): Number of points in the wealth grid.
        """

        super().__init__()  # Initialize the Model superclass

        # --- Model Parameters ---
        self.num_agents = sum(population_composition.values())
        self.interest_rate = interest_rate  # Constant gross interest rate
        self.sigma = sigma  # inv. of intertemporal substitution
        self.max_wealth = 1000001  # Upper bound for the wealth grid
        self.borrowing_limit = 0    # Lower bound for wealth (no borrowing)
        self.wealth_dist = wealth_dist  # Initial wealth distribution
        self.num_wealth_points = num_wealth_points
        min_grid_wealth = 1e-6 

        # --- Mesa Components ---
        # Grid is not strictly necessary for a single agent but is kept for
        # compatibility with potential future multi-agent extensions.
        self.grid = MultiGrid(10, 10, True)  # A 10x10 grid
        self.schedule = RandomActivation(self)  # Random activation scheduler

        if self.max_wealth <= min_grid_wealth:
            raise ValueError(f"max_wealth ({self.max_wealth}) must be greater than min_grid_wealth ({min_grid_wealth})")

        # --- Create the Wealth Grid ---
        # A discrete set of wealth levels used for value function iteration.
        self.wealth_grid = np.geomspace(min_grid_wealth, self.max_wealth, self.num_wealth_points)
        print(f"Model Initialized with a GEOMETRICALLY SPACED wealth_grid of {self.num_wealth_points} points from {self.wealth_grid[0]:.2e} to {self.wealth_grid[-1]:.2e}.", flush=True)



        # --- Create the Agent ---
        self.create_agents(population_composition)    # Create and add the agent

        # --- Data Collection ---
        # Define consumption calculation carefully based on when savings are decided
        # Consumption in step t depends on wealth at start of t (which is savings from t-1)
        # and savings chosen in step t (which becomes wealth at start of t+1).
        def get_consumption(agent):
            # Ensure previous_wealth is not None (can happen before first step completes fully)
            if agent.previous_wealth is None:
                return 0 # Or some other placeholder like np.nan
            consumption = agent.model.interest_rate * agent.previous_wealth - agent.savings
            return max(consumption, 1e-9) # Ensure non-negative

        def get_utility(agent):
             consumption = get_consumption(agent)
             # Handle cases where consumption might lead to invalid utility (e.g., log(0))
             utility_val = agent.utility(consumption)
             if not np.isfinite(utility_val):
                 # print(f"Warning: Non-finite utility calculated ({utility_val}) for consumption {consumption}. Returning NaN.")
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
            },
            agent_reporters={
                "Wealth": "wealth", # Wealth at the *end* of the step (i.e., next period's start)
                "Savings": "savings", # Savings chosen *during* the step
                "Consumption": get_consumption, # Consumption *during* the step
                "Total Resources": lambda a: a.previous_wealth * a.model.interest_rate if a.previous_wealth is not None else 0,
                "R_star": "R_star",
                "Interest_Rate": lambda a: a.model.interest_rate,
                "Utility": get_utility, # Utility from consumption *during* the step
                "Previous_Wealth": "previous_wealth", 
                "Profile": "profile_name"
            }
        )
        print("Model initialized", flush=True)

    def create_agents(self, population_composition):
        """
        Creates a population of SavingAgents based on the specified composition
        and adds them to the model's schedule.

        Args:
            population_composition (dict): A dictionary where keys are profile
                names (str) and values are the number of agents (int) to create
                for that profile.
        """
        print("Creating agent population...")
        # The global agent_profiles dictionary is used to get parameters for each profile name
        global agent_profiles

        # Loop through the population_composition dictionary to create agents for each profile
        for profile_name, count in population_composition.items():
            if profile_name not in agent_profiles:
                print(f"Warning: Profile '{profile_name}' not found in agent_profiles. Skipping.")
                continue

            print(f"  Creating {count} agent(s) with profile: '{profile_name}'")
            profile_params = agent_profiles[profile_name] # Get the parameters for this profile

            # Create the specified number of agents for the current profile
            for i in range(count):
                # --- Determine Initial Wealth for each agent ---
                rand_num = random.random()
                cumulative_prob = 0
                init_wealth = 0
                for prob, wealth_range in self.wealth_dist:
                    cumulative_prob += prob
                    if rand_num <= cumulative_prob:
                        init_wealth = random.randint(wealth_range[0], wealth_range[1])
                        break
                else:
                    init_wealth = random.randint(max(1, self.wealth_dist[-1][1][0]), self.wealth_dist[-1][1][1])

                agent_vfi_iterations = profile_params.get("vfi_iterations", 50)

                # --- Create and Add the Agent ---
                agent = SavingAgent(
                    unique_id=self.next_id(), # Mesa handles unique IDs
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

        print(f"Total agents created and added to schedule: {len(self.schedule.agents)}")
        


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
    "planner": {"beta": 0.97, "delta": 0.96,"vfi_iterations": 150},  # Higher beta = less present bias, higer delta = more patient ° default = "beta": 0.8, "delta": 0.98,"vfi_iterations": 60
    "moderate": {"beta": 0.90, "delta": 0.91, "vfi_iterations": 150},  # Base values
    "procrastinator": {"beta": 0.73, "delta": 0.95, "vfi_iterations": 150},  # low beta = more present bias, high delta = more patient, values also future consumption
    "inverse procrastinator": {"beta": 0.96, "delta": 0.85, "vfi_iterations": 150},
    "impulsive": {"beta": 0.60, "delta": 0.80, "vfi_iterations": 150},  # lower beta = more present bias, lower delta = less patient
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
    "planner": 5,
    "moderate": 10,
    "impulsive": 8
}

# Define the conditions to iterate over
interest_rates_to_test = [1.02, 1.05, 1.10, 1.30]
SIMULATION_STEPS = 24

# --- 2. Setup Output Directory ---
output_dir_csv = "output_csv"
os.makedirs(output_dir_csv, exist_ok=True)
output_dir_text = "output_text"
os.makedirs(output_dir_text, exist_ok=True)
log_filepath = os.path.join(output_dir_text, "simulation_run_log.txt")

print(f"Starting simulation. All detailed output will be saved to: {log_filepath}")

# Use a 'with' block to handle the file and stdout redirection
with open(log_filepath, "w") as log_file:
    original_stdout = sys.stdout
    sys.stdout = log_file # All subsequent 'print' statements go to the log_file
    print("\n" + "="*50)
    print("ALL SIMULATIONS COMPLETE...") 

    for rate in interest_rates_to_test:
        print(f"\n--- Running Simulation for Interest Rate (R) = {rate} ---")
        
        # Create a fresh model instance for this condition
        model = SavingModel(
            population_composition=population_to_simulate,
            interest_rate=rate,
            sigma=sigma,
            wealth_dist=wealth_dist,
            num_wealth_points=100 # Size of the wealth_grid
        )
        
        # Run the model
        for i in range(SIMULATION_STEPS):
            model.step()
            
        print(f"--- Simulation Complete. Exporting data... ---")

        # Retrieve data from the datacollector
        agent_data = model.datacollector.get_agent_vars_dataframe()
        model_data = model.datacollector.get_model_vars_dataframe()

        print(f"DEBUG: Columns in agent_data for R={rate} are: {agent_data.columns}")

        print("--- First 15 Profile Values in the DataFrame ---")
        print(agent_data[['Profile']].head(15))
        print("---------------------------------------------")

        print("--- Exporting V and g functions for sample agents... ---")
        
        # Create a sub-directory for this detailed data
        output_dir_v_g = os.path.join(output_dir_csv, "v_g_functions")
        os.makedirs(output_dir_v_g, exist_ok=True)

        # Identify one agent from each profile to save their functions
        agents_to_save = {}

        for profile in population_to_simulate.keys():
            print(f"Searching for agents with profile: '{profile}'") 
            
            # This is the line that is failing
            filtered_agents = agent_data[agent_data['Profile'] == profile]
            
            if filtered_agents.empty:
                print(f"  > WARNING: No agents found for profile '{profile}'. Skipping.")
                continue # Skip to the next profile in the loop
                
            first_agent_of_profile = filtered_agents.index.get_level_values('AgentID')[0]
            agents_to_save[profile] = first_agent_of_profile
        
        print(f"Found sample agents to save: {agents_to_save}") 

        for agent in model.schedule.agents:
            if agent.unique_id in agents_to_save.values():
                if agent.value_function_calculated:
                    v_g_base_filename = f"R_{rate}_agent_{agent.unique_id}_{agent.profile_name}"
                    v_func_path = os.path.join(output_dir_v_g, f"{v_g_base_filename}_value_function.npy")
                    g_func_path = os.path.join(output_dir_v_g, f"{v_g_base_filename}_policy_function.npy")
                    
                    np.save(v_func_path, agent.V)
                    np.save(g_func_path, agent.g)
                    print(f"  > Saved V and g for Agent {agent.unique_id} ({agent.profile_name})")
        
        # Define descriptive filenames
        base_filename = f"R_{rate}_pop_{len(model.schedule.agents)}agents_steps_{SIMULATION_STEPS}"
        agent_data_filepath = os.path.join(output_dir_csv, f"{base_filename}_agent_data.csv")
        model_data_filepath = os.path.join(output_dir_csv, f"{base_filename}_model_data.csv")
        
        # Save data to CSV files
        agent_data.to_csv(agent_data_filepath)
        model_data.to_csv(model_data_filepath)
        
        print(f"Successfully saved Agent Data to: {agent_data_filepath}")
        print(f"Successfully saved Model Data to: {model_data_filepath}")

    print("\n" + "="*50)
    print("ALL SIMULATIONS COMPLETE. ALL DATA EXPORTED.")
    print(f"Output files are in the '{output_dir_csv}' directory.")
    print("="*50 + "\n")

# --- Restore console output ---
sys.stdout = original_stdout

# This print statement will appear in your console
print("Process finished.")
print(f"All data saved to '{output_dir_csv}'.")
print(f"Full simulation log saved to '{log_filepath}'.")