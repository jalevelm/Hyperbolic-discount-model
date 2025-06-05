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
    def __init__(self, unique_id, model, beta, delta, init_wealth, sigma, borrowing_limit, iterations=50):
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
            print(f"Agent {self.unique_id}: First Step - Beta: {self.beta}, Delta: {self.delta}, R_star: {self.R_star}", flush=True)
        
        # --- Debugging Prints (Start of Step) ---
        print(f"Agent {self.unique_id}: Step start. Wealth: {self.wealth}, Previous Savings: {self.previous_savings}", flush=True)
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
    
    def __init__(self, agent_profile, interest_rate, sigma, wealth_dist, num_wealth_points=100):
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
        self.num_agents = 1  #Number of agents
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
        self.create_agent(agent_profile)    # Create and add the agent

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
            agent_reporters={
                "Wealth": "wealth", # Wealth at the *end* of the step (i.e., next period's start)
                "Savings": "savings", # Savings chosen *during* the step
                "Consumption": get_consumption, # Consumption *during* the step
                "R_star": "R_star",
                "Interest_Rate": lambda a: a.model.interest_rate,
                "Utility": get_utility, # Utility from consumption *during* the step
                "Previous_Wealth": "previous_wealth" 
            }
        )
        print("Model initialized", flush=True)

    def create_agent(self, profile):
        """
        Creates a single SavingAgent and adds it to the model.

        Args:
            profile (dict): A dictionary containing the agent's 'beta' and
                'delta' values, defining its hyperbolic discounting preferences.
        """
        print(f"Creating agent with profile: {profile}", flush=True)

        # --- Determine Initial Wealth (based on wealth_dist) ---
        rand_num = random.random()  # Generate a random number between 0 and 1
        cumulative_prob = 0
        init_wealth = 0
        # Iterate through the wealth distribution to determine the agent's
        # initial wealth based on the defined probabilities.
        for prob, wealth_range in self.wealth_dist:
            cumulative_prob += prob
            if rand_num <= cumulative_prob:
                init_wealth = random.randint(wealth_range[0], wealth_range[1])
                break
        else:
            # If no range is selected (shouldn't happen with a proper
            # distribution summing to 1), assign wealth from the last range.
            init_wealth = random.randint(max(1, self.wealth_dist[-1][1][0]), self.wealth_dist[-1][1][1])

        agent_vfi_iterations = profile.get("vfi_iterations", 50)
        print(f"  Passing vfi_iterations={agent_vfi_iterations} to SavingAgent constructor.", flush=True) # Add this print for debugging

    # --- Create and Add the Agent, passing the agent_vfi_iterations ---
        agent = SavingAgent(self.next_id(), # Or your agent ID generation, e.g., 0 if only one agent
                        self,
                        profile["beta"],
                        profile["delta"],
                        init_wealth,
                        self.sigma, # Assuming sigma is a model attribute or passed correctly
                        self.borrowing_limit,
                        iterations=agent_vfi_iterations) # <-- MAKE SURE THIS IS PASSED

        self.schedule.add(agent)
        # Assuming grid placement is still relevant, even if minimal
        # self.grid.place_agent(agent, (0,0)) # Or however you handle agent placement
        print(f"Agent created with ID {agent.unique_id} and added to schedule. Initial wealth: {init_wealth}", flush=True)
        


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

# ---------------------------------------------------------- Model run block ------------------------------------------------------------------------


# --- Define Agent Profiles ---
agent_profiles = {
    "planner": {"beta": 0.97, "delta": 0.96,"vfi_iterations": 1500},  # Higher beta = less present bias, higer delta = more patient ° default = "beta": 0.8, "delta": 0.98,"vfi_iterations": 60
    "moderate": {"beta": 0.90, "delta": 0.91, "vfi_iterations": 1500},  # Base values
    "procrastinator": {"beta": 0.73, "delta": 0.95, "vfi_iterations": 1500},  # low beta = more present bias, high delta = more patient, values also future consumption
    "inverse procrastinator": {"beta": 0.96, "delta": 0.85, "vfi_iterations": 1500},
    "impulsive": {"beta": 0.60, "delta": 0.80, "vfi_iterations": 1500},  # lower beta = more present bias, lower delta = less patient
}

sigma = 0.4387 # inverse of IES

wealth_dist = [
    (0.4152, (0, 9999)),  # Less than 10,000 USD
    (0.4772, (10000, 99999)),  # Between 10,000 and 100,000 USD
    (0.1031, (100000, 999999)),  # Between 100,000 and 1,000,000 USD
    (0.0045, (1000000, 1000001))  # More than 1,000,000 USD 
]

'''
# --- Define Interest Rates to Simulate ---
interest_rates_to_simulate = [1.01, 1.10, 1.20, 1.30] # interest rates (Gross Rate R)

# --- Define Simulation Steps ---
simulation_steps = 5 # Number of steps per simulation run

# --- Setup Output Directories ---
output_dir_text = "output_text" # Directory for text log
output_dir_plots = "output_plots" # Directory for plot images
os.makedirs(output_dir_text, exist_ok=True)
os.makedirs(output_dir_plots, exist_ok=True)
output_file_path = os.path.join(output_dir_text, "simulation_output.txt")

# --- Data Storage Structure ---
# Dictionary: { profile_name -> { interest_rate -> pandas_dataframe } }
results_data = {profile_name: {} for profile_name in agent_profiles.keys()}
# Store agent objects temporarily if needed for V/g plots
agent_instances = {profile_name: {} for profile_name in agent_profiles.keys()}


# --- Redirect Output to File ---
print(f"Redirecting simulation stdout log to: {output_file_path}")
with open(output_file_path, "w") as output_file:
    original_stdout = sys.stdout # Store original stdout
    sys.stdout = output_file  # Redirect standard output to the file

# --------------------------------------------------------
# --- Phase 1: Run Simulations & Collect Data ----------
# --------------------------------------------------------
    print("="*70)
    print("Starting Simulation Runs", flush=True)
    print("="*70)
'''
#---------------------------------------------------------PROFILING BLOCK----------------------------------------------------------------------------------
PROFILES_TO_ITERATE = list(agent_profiles.keys()) 
INTEREST_RATES_TO_ITERATE = [1.02, 1.10, 1.30] 

# These parameters will be common for all profiling runs in this iteration
NUM_WEALTH_POINTS_FOR_PROFILING = 5000 # Size of the wealth_grid for these profiling runs
SIMULATION_STEPS_PER_RUN = 12 # Number of steps for each individual simulation run (VFI usually happens in step 1)


# --- Setup Output Directories (same as before) ---
output_dir_text = "output_text"
os.makedirs(output_dir_text, exist_ok=True)
# output_dir_plots is not strictly needed for profiling VFI but doesn't hurt
output_dir_plots = "output_plots"
os.makedirs(output_dir_plots, exist_ok=True)
# Use a distinct log file for profiling output
iterative_profiling_log_path = os.path.join(output_dir_text, "simulation_output_ITERATIVE_PROFILING.txt")

# --- Redirect Output to File  ---
print(f"Redirecting simulation stdout log to: {iterative_profiling_log_path}")
with open(iterative_profiling_log_path, "w") as output_file: 
    original_stdout = sys.stdout
    sys.stdout = output_file # Redirect standard output

    # --- START OF PROFILING LOGIC ---
    print("="*70, flush=True)
    print(f"Starting ITERATIVE PROFILING Runs", flush=True)
    print(f"Target Profiles: {PROFILES_TO_ITERATE}", flush=True)
    print(f"Target Interest Rates: {INTEREST_RATES_TO_ITERATE}", flush=True)
    print(f"Wealth Grid Points for each run: {NUM_WEALTH_POINTS_FOR_PROFILING}", flush=True)
    print(f"Simulation Steps for each run: {SIMULATION_STEPS_PER_RUN}", flush=True)
    print("="*70, flush=True)


    for current_profile_name in PROFILES_TO_ITERATE:
        if current_profile_name not in agent_profiles:
            print(f"\nXXXXX SKIPPING: Profile '{current_profile_name}' not found in agent_profiles. XXXXX\n", flush=True)
            continue
        
        profile_config_for_run = agent_profiles[current_profile_name]
        expected_vfi_iterations = profile_config_for_run.get("vfi_iterations", "DEFAULT(50)")

        # Loop through each interest rate for the current profile
        for current_interest_rate in INTEREST_RATES_TO_ITERATE:
            print(f"\n" + "="*70, flush=True)
            print(f"PROFILING: Profile='{current_profile_name}', R={current_interest_rate:.2f}", flush=True)
            print(f"  Agent VFI Iterations expected: {expected_vfi_iterations}", flush=True)
            print("="*70, flush=True)

            profiler_for_run = cProfile.Profile() # Initialize a new profiler for each combination
            model_for_run = None # Initialize model to None for each run
            run_start_time = time.time()

            try:
                profiler_for_run.enable()

                model_for_run = SavingModel(
                    profile_config_for_run,
                    current_interest_rate,
                    sigma, # Global sigma
                    wealth_dist, # Global wealth_dist
                    num_wealth_points=NUM_WEALTH_POINTS_FOR_PROFILING
                )

                for i in range(SIMULATION_STEPS_PER_RUN):
                    # print(f"  Model Step {i+1}/{SIMULATION_STEPS_PER_RUN}", flush=True) # Optional per-step log
                    model_for_run.step()

            except Exception as e:
                output_file.write(f"\n!!!!!! ERROR during PROFILING run for Profile: {current_profile_name}, R={current_interest_rate:.2f} !!!!!!\n")
                output_file.write(f"Error type: {type(e).__name__}\n")
                output_file.write(f"Error message: {str(e)}\n")
                import traceback
                output_file.write("\nTraceback:\n")
                traceback.print_exc(file=output_file)
                output_file.write("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                print(f"ERROR encountered for {current_profile_name}, R={current_interest_rate}, see log.", flush=True)
            finally:
                if 'profiler_for_run' in locals(): # Check if profiler was created for this run
                    profiler_for_run.disable()
                    # print("Profiler disabled for this run.", flush=True) # Optional

                run_end_time = time.time()
                print(f"Simulation (including VFI) for Profile='{current_profile_name}', R={current_interest_rate:.2f} completed/stopped in {run_end_time - run_start_time:.4f} seconds.", flush=True)

                if model_for_run: # Only print stats and plots if model was successfully created
                    output_file.write(f"\n\n----- CPROFILE STATS FOR VFI -----\n")
                    output_file.write(f"Profile: {current_profile_name}, R={current_interest_rate:.2f}, GridPoints: {NUM_WEALTH_POINTS_FOR_PROFILING}\n")
                    stats_for_run = pstats.Stats(profiler_for_run, stream=output_file).sort_stats('cumulative')
                    stats_for_run.print_stats(50) # Print top 50 functions by cumulative time
                    output_file.write(f"----- END CPROFILE STATS -----\n\n")
                    print("Profiling stats for this run written to log file.", flush=True)

                    if hasattr(model_for_run.schedule, 'agents') and model_for_run.schedule.agents: # Check if agents list exists and is not empty
                        if hasattr(model_for_run.schedule.agents[0], 'V_snapshots'): # Now safe to access agent
                            agent_for_plots = model_for_run.schedule.agents[0]
                            wealth_grid_for_plots = model_for_run.wealth_grid
                        
                        print("Generating V and g snapshot plots for this run...", flush=True)
                        # V snapshots plot
                        if len(agent_for_plots.V_snapshots) > 0:
                            plt.figure(figsize=(12, 6))
                            for iter_num, v_snap in agent_for_plots.V_snapshots.items():
                                plt.plot(wealth_grid_for_plots, v_snap, label=f'V iter {iter_num}')
                            plt.title(f'Value Function (V) Snapshots\nProfile: {current_profile_name}, Beta={agent_for_plots.beta:.2f}, Delta={agent_for_plots.delta:.2f}, R={current_interest_rate:.2f}, Grid={len(wealth_grid_for_plots)}')
                            plt.xlabel('Wealth (k)')
                            plt.ylabel('Value V(k)')
                            plt.legend()
                            plt.grid(True)
                            plot_filename_v_snap = os.path.join(output_dir_plots, f"V_snapshots_{current_profile_name}_R{current_interest_rate:.2f}.png")
                            try:
                                plt.savefig(plot_filename_v_snap)
                                print(f"Saved V snapshots plot: {plot_filename_v_snap}", flush=True)
                            except Exception as e_plot:
                                print(f"Error saving V snapshots plot: {e_plot}", flush=True)
                            plt.close() # Close the figure

                        # G snapshots plot
                        if len(agent_for_plots.g_snapshots) > 0:
                            plt.figure(figsize=(12, 6))
                            for iter_num, g_snap in agent_for_plots.g_snapshots.items():
                                plt.plot(wealth_grid_for_plots, g_snap, label=f'g iter {iter_num} (Savings k\')')
                            plt.plot(wealth_grid_for_plots, wealth_grid_for_plots, 'k--', label='k\' = k (45-deg line)', alpha=0.7)
                            plt.title(f'Policy Function (g) Snapshots\nProfile: {current_profile_name}, Beta={agent_for_plots.beta:.2f}, Delta={agent_for_plots.delta:.2f}, R={current_interest_rate:.2f}, Grid={len(wealth_grid_for_plots)}')
                            plt.xlabel('Current Wealth (k)')
                            plt.ylabel('Next Period Wealth (k\') = Savings')
                            plt.legend()
                            plt.grid(True)
                            plot_filename_g_snap = os.path.join(output_dir_plots, f"g_snapshots_{current_profile_name}_R{current_interest_rate:.2f}.png")
                            try:
                                plt.savefig(plot_filename_g_snap)
                                print(f"Saved g snapshots plot: {plot_filename_g_snap}", flush=True)
                            except Exception as e_plot:
                                print(f"Error saving g snapshots plot: {e_plot}", flush=True)
                            plt.close() # Close the figure
                        
                        # --- Additional Time Series Plots ---
                        print("Generating additional time series plots for this run...", flush=True)
                        num_steps_completed = agent_for_plots.step_count
                        if num_steps_completed > 0:
                            time_axis_consum_util = range(1, num_steps_completed + 1)
                            time_axis_wealth = range(num_steps_completed + 1)

                            # Consumption Plot
                            plt.figure(figsize=(10, 6))
                            plt.plot(time_axis_consum_util, agent_for_plots.consumption_history, marker='o', linestyle='-', label="Consumption")
                            plt.title(f'Agent Consumption Over Time\nProfile: {current_profile_name}, Beta={agent_for_plots.beta:.2f}, Delta={agent_for_plots.delta:.2f}, R={current_interest_rate:.2f}')
                            plt.xlabel('Time Step'); plt.ylabel('Consumption')
                            if num_steps_completed <= 20: plt.xticks(range(1, num_steps_completed + 1))
                            plt.grid(True); plt.legend()
                            plot_filename_consum = os.path.join(output_dir_plots, f"TimeSeries_Consumption_{current_profile_name}_R{current_interest_rate:.2f}.png")
                            try: plt.savefig(plot_filename_consum); print(f"Saved plot: {plot_filename_consum}", flush=True)
                            except Exception as e: print(f"Error saving plot {plot_filename_consum}: {e}", flush=True)
                            plt.close()

                            # Wealth Plot
                            plt.figure(figsize=(10, 6))
                            plt.plot(time_axis_wealth, agent_for_plots.wealth_history, marker='o', linestyle='-', label="Wealth")
                            plt.title(f'Agent Wealth Over Time\nProfile: {current_profile_name}, Beta={agent_for_plots.beta:.2f}, Delta={agent_for_plots.delta:.2f}, R={current_interest_rate:.2f}')
                            plt.xlabel('Time Step (0 = Initial Wealth)'); plt.ylabel('Wealth')
                            if num_steps_completed <= 20: plt.xticks(range(num_steps_completed + 1))
                            plt.grid(True); plt.legend()
                            plot_filename_wealth = os.path.join(output_dir_plots, f"TimeSeries_Wealth_{current_profile_name}_R{current_interest_rate:.2f}.png")
                            try: plt.savefig(plot_filename_wealth); print(f"Saved plot: {plot_filename_wealth}", flush=True)
                            except Exception as e: print(f"Error saving plot {plot_filename_wealth}: {e}", flush=True)
                            plt.close()

                            # Utility Plot
                            plt.figure(figsize=(10, 6))
                            plt.plot(time_axis_consum_util, agent_for_plots.utility_history, marker='o', linestyle='-', label="Utility")
                            plt.title(f'Agent Utility Over Time\nProfile: {current_profile_name}, Beta={agent_for_plots.beta:.2f}, Delta={agent_for_plots.delta:.2f}, R={current_interest_rate:.2f}')
                            plt.xlabel('Time Step'); plt.ylabel('Utility')
                            if num_steps_completed <= 20: plt.xticks(range(1, num_steps_completed + 1))
                            plt.grid(True); plt.legend()
                            plot_filename_utility = os.path.join(output_dir_plots, f"TimeSeries_Utility_{current_profile_name}_R{current_interest_rate:.2f}.png")
                            try: plt.savefig(plot_filename_utility); print(f"Saved plot: {plot_filename_utility}", flush=True)
                            except Exception as e: print(f"Error saving plot {plot_filename_utility}: {e}", flush=True)
                            plt.close()
                        else:
                            print("No agent steps completed, skipping time series plots for this run.", flush=True)
                    else:
                        print("Agent data for plots not found for this run.", flush=True)
                else:
                    print("Model not successfully initialized, skipping stats and plots for this run.", flush=True)
                
                plt.close('all') # Close any remaining figures for this iteration to save memory

            print(f"----- COMPLETED RUN for Profile='{current_profile_name}', R={current_interest_rate:.2f} -----\n", flush=True)
            # End of interest rate loop
        # End of profile loop

    print("\n" + "="*70, flush=True)
    print("ALL ITERATIVE PROFILING Runs Completed.", flush=True)
    print("="*70 + "\n", flush=True)

# --- Restore Standard Output ---
sys.stdout = original_stdout
print(f"\nIterative profiling stdout log saved to: {iterative_profiling_log_path}")
print(f"All plots saved to directory: {output_dir_plots}")
'''
    # Loop through agent profiles first
    
    for profile_name, profile in agent_profiles.items():
        print(f"\n===== Running Simulations for Profile: {profile_name} =====", flush=True)
        print(f"Profile parameters: Beta={profile['beta']}, Delta={profile['delta']}", flush=True)

        # Then loop through interest rates for this profile
        for interest_rate in interest_rates_to_simulate:
            print(f"\n--- Interest Rate (R): {interest_rate:.2f} ---", flush=True)
            start_time_sim = time.time()

            # --- Initialize and Run Model ---
            try:
                model = SavingModel(profile, interest_rate, sigma, wealth_dist)
                # Run the simulation for the specified number of steps
                for i in range(simulation_steps):
                    # print(f"  Step {i+1}/{simulation_steps}") # Keep commented for brevity unless needed
                    model.step()

                # --- Data Retrieval and Storage ---
                agent_data = model.datacollector.get_agent_vars_dataframe()

                # Store the collected data (pandas DataFrame)
                results_data[profile_name][interest_rate] = agent_data
                # Store the agent instance (needed for V and g plots)
                # Assumes only one agent with unique_id 0 exists
                agent_instances[profile_name][interest_rate] = model.schedule.agents[0]

                end_time_sim = time.time()
                print(f"Simulation for Profile '{profile_name}' at R={interest_rate:.2f} completed in {end_time_sim - start_time_sim:.4f} seconds.", flush=True) 
                if agent_data is not None and not agent_data.empty and 'Wealth' in agent_data.columns:
                     print(f"  Final Wealth: {agent_data['Wealth'].iloc[-1]:.2f}", flush=True)
                else:
                     print(f"  Final Wealth: N/A (Simulation Error or No Data)", flush=True)


            except Exception as e:
                 print(f"\n!!!!!! ERROR during simulation for Profile: {profile_name}, R={interest_rate} !!!!!!", flush=True)
                 print(f"Error type: {type(e).__name__}", flush=True)
                 print(f"Error message: {e}", flush=True)
                 import traceback
                 print("Traceback:")
                 traceback.print_exc(file=output_file) # Print traceback to the log file
                 # Store None to indicate failure for this run
                 results_data[profile_name][interest_rate] = None
                 agent_instances[profile_name][interest_rate] = None
                 print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


    print("\n" + "="*70)
    print("All Simulation Runs Completed", flush=True)
    print("="*70 + "\n")


# --------------------------------------------------------
# --- Phase 2: Generate Plots --------------------------
# --------------------------------------------------------
    print("="*70)
    print("Generating Plots")
    print("="*70)

    # --- Plot 1: Value and Policy Functions (One per profile/rate) ---
    # [ This section remains unchanged - keep as is ]
    print("\n--- Generating Value/Policy Function Plots ---")
    for profile_name in agent_profiles.keys():
        for interest_rate in interest_rates_to_simulate:
            agent = agent_instances[profile_name].get(interest_rate) 
            # Check if agent exists and VFI was completed
            if agent and agent.value_function_calculated:
                model = agent.model # Get model reference from agent for wealth_grid
                plt.figure(figsize=(12, 5)) # Slightly wider figure

                # Plot Value Function
                plt.subplot(1, 2, 1)
                plt.plot(model.wealth_grid, agent.V, label=f'V(k)')
                plt.title(f"Value Function V(k)\nProfile: {profile_name}, R={interest_rate:.2f}")
                plt.xlabel("Wealth (k)")
                plt.ylabel("Value V(k)")
                plt.grid(True, linestyle=':', alpha=0.6)
                plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) # Use sci notation if numbers are large

                # Plot Policy Function
                plt.subplot(1, 2, 2)
                plt.plot(model.wealth_grid, agent.g, label='g(k) - Savings Policy')
                # Plot 45-degree line for reference (k_next = k)
                plt.plot(model.wealth_grid, model.wealth_grid, color='red', linestyle='--', linewidth=1, label="k'=k (45-degree line)")
                plt.title(f"Policy Function g(k) (Savings)\nProfile: {profile_name}, R={interest_rate:.2f}")
                plt.xlabel("Current Wealth (k)")
                plt.ylabel("Next Period Wealth (k') = Savings")
                plt.legend()
                plt.grid(True, linestyle=':', alpha=0.6)
                plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

                plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout slightly if titles overlap
                # Saving plots to the specified directory
                plot_filename_vg = os.path.join(output_dir_plots, f"{profile_name}_R{interest_rate:.2f}_ValuePolicy.png")
                try:
                    plt.savefig(plot_filename_vg)
                    # print(f"Saved Value/Policy plot: {plot_filename_vg}") # Keep commented unless needed
                except Exception as e:
                    print(f"Error saving plot {plot_filename_vg}: {e}")
                plt.close() # Close the figure to free memory
            elif agent is None:
                 # print(f"Skipping Value/Policy plot for {profile_name}, R={interest_rate:.2f} (Simulation Error)") # Keep commented unless needed
                 pass # Pass silently
            #else:
                 # print(f"Skipping Value/Policy plot for {profile_name}, R={interest_rate:.2f} (VFI not run or agent missing)") # Keep commented unless needed


    # --- Plot 2: Combined Time Series Plots (One per interest rate) ---
    print("\n--- Generating Combined Time Series Plots (Now 3x2 Layout) ---")
    # Use a color cycle for different profiles
    colors = plt.cm.viridis(np.linspace(0, 1, len(agent_profiles))) # Example color map
    profile_colors = {name: colors[i] for i, name in enumerate(agent_profiles.keys())}

    # Loop through each interest rate to create a combined plot
    for interest_rate in interest_rates_to_simulate:
        print(f"  Generating combined plot for R = {interest_rate:.2f}")
        # *****Update layout to 3 rows, 2 columns *****
        fig, axes = plt.subplots(3, 2, figsize=(14, 18)) # Adjust figsize if needed
        fig.suptitle(f"Simulation Time Series Results (R = {interest_rate:.2f})", fontsize=16)
        axes = axes.flatten() # Flatten axes array for easier indexing (now 6 axes: 0-5)

        plot_successful = False # Flag to check if any data was plotted for this rate
        # *****Track plotted profiles for 6 subplots *****
        plotted_profiles_on_subplot = [set() for _ in range(6)]

        # --- Data storage for the bar chart for this interest rate ---
        profile_names_for_bar = []
        summed_savings_for_bar = []
        # ----------------------------------------------------------

        # Loop through each profile to plot its data on the current figure
        for profile_index, profile_name in enumerate(agent_profiles.keys()):
            # Retrieve the dataframe for this profile and interest rate
            agent_data = results_data[profile_name].get(interest_rate) # Use .get for safety

            # Check if data exists (simulation might have failed)
            if agent_data is not None and not agent_data.empty:
                color = profile_colors[profile_name] # Get color for this profile

                # Explicitly convert data before plotting and handle potential errors
                try:
                    # Get Step values from index level 0
                    x_step_data = agent_data.index.get_level_values(0)

                    # Convert relevant Y-data columns to numeric, raising error on failure
                    y_prev_wealth_data = pd.to_numeric(agent_data["Previous_Wealth"], errors='raise')
                    y_savings_data = pd.to_numeric(agent_data["Savings"], errors='raise')
                    y_consumption_data = pd.to_numeric(agent_data["Consumption"], errors='raise')
                    y_utility_data = pd.to_numeric(agent_data["Utility"], errors='raise')

                    # Calculate cumulative savings
                    y_cumulative_savings = y_savings_data.cumsum()

                    # --- Subplot 1 (axes[0]): Wealth & Savings ---
                    axes[0].plot(x_step_data, y_prev_wealth_data, marker='o', linestyle='-', color=color, label=f"{profile_name} Wealth") # Simplified label
                    axes[0].plot(x_step_data, y_savings_data, marker='x', linestyle='--', color=color, label=f"{profile_name} Savings") # Simplified label
                    axes[0].set_title("Wealth (Start) and Savings (Decision)")
                    axes[0].set_xlabel("Time Step")
                    axes[0].set_ylabel("Amount ($)")
                    axes[0].grid(True, linestyle=':', alpha=0.6)
                    axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                    plotted_profiles_on_subplot[0].add(profile_name) # Mark as plotted

                    # --- Subplot 2 (axes[1]): Consumption ---
                    axes[1].plot(x_step_data, y_consumption_data, marker='o', linestyle='-', color=color, label=f"{profile_name}")
                    axes[1].set_title("Consumption")
                    axes[1].set_xlabel("Time Step")
                    axes[1].set_ylabel("Amount ($)")
                    axes[1].grid(True, linestyle=':', alpha=0.6)
                    axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                    plotted_profiles_on_subplot[1].add(profile_name)

                    # --- Subplot 3 (axes[2]): Utility ---
                    axes[2].plot(x_step_data, y_utility_data, marker='o', linestyle='-', color=color, label=f"{profile_name}")
                    axes[2].set_title("Period Utility")
                    axes[2].set_xlabel("Time Step")
                    axes[2].set_ylabel("Utility Value")
                    axes[2].grid(True, linestyle=':', alpha=0.6)
                    # Utility might not need sci notation unless sigma is very high/low
                    plotted_profiles_on_subplot[2].add(profile_name)

                    # ***** NEW: Subplot 4 (axes[3]): Cumulative Savings *****
                    axes[3].plot(x_step_data, y_cumulative_savings, marker='o', linestyle='-', color=color, label=f"{profile_name}")
                    axes[3].set_title("Cumulative Savings")
                    axes[3].set_xlabel("Time Step")
                    axes[3].set_ylabel("Total Savings Accumulated ($)")
                    axes[3].grid(True, linestyle=':', alpha=0.6)
                    axes[3].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                    plotted_profiles_on_subplot[3].add(profile_name)

                    # ***** CHANGE: Subplot 5 (axes[4]): Wealth (Start) vs Savings *****
                    axes[4].plot(y_prev_wealth_data, y_savings_data, marker='o', linestyle='-', color=color, label=f"{profile_name}")
                    axes[4].set_title("Wealth (Start) vs. Savings")
                    axes[4].set_xlabel("Wealth at Start of Step ($)")
                    axes[4].set_ylabel("Savings Chosen in Step ($)")
                    axes[4].grid(True, linestyle=':', alpha=0.6)
                    axes[4].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                    axes[4].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                    # Add 45 degree line for reference if scales allow (only label once per plot)
                    if axes[4].has_data(): # Check if limits are valid before drawing line
                        is_first_profile_on_subplot5 = not plotted_profiles_on_subplot[4] # Check before adding
                        plotted_profiles_on_subplot[4].add(profile_name) # Mark profile as plotted *after* check
                        try: # Getting limits might fail if data is weird
                            min_val_x, max_val_x = axes[4].get_xlim()
                            min_val_y, max_val_y = axes[4].get_ylim()
                            min_val = min(min_val_x, min_val_y)
                            max_val = max(max_val_x, max_val_y)
                            if max_val > min_val: # Avoid plotting line if range is zero or invalid
                                axes[4].plot([min_val, max_val], [min_val, max_val], color='grey', linestyle=':', linewidth=1, label="Savings = Wealth" if is_first_profile_on_subplot5 else None)
                        except Exception as lim_err:
                            print(f"    Warning: Could not draw 45-degree line on subplot 5 ({lim_err})")


                    # --- Subplot 6 (axes[5]): Empty ---
                    # Leave axes[5] empty or hide it
                    axes[5].set_visible(False) # Hide the last unused subplot
                    plotted_profiles_on_subplot[5].add(profile_name) # Mark anyway to keep loop consistent


                    # --- Data for Bar Chart ---
                    # Calculate sum of savings for this profile/rate
                    total_s = y_savings_data.sum()
                    profile_names_for_bar.append(profile_name)
                    summed_savings_for_bar.append(total_s)
                    # --------------------------

                    plot_successful = True # Mark that at least one profile plotted successfully for this rate

                except Exception as e:
                    # Error during Y-data conversion or plotting for this profile
                    print(f"    ERROR during Y-data conversion or plotting for {profile_name}, R={interest_rate}: {e}", flush=True)
                    # Print problematic Y-data
                    if 'Previous_Wealth' in agent_data.columns: print(f"    Problematic Previous_Wealth ({type(agent_data['Previous_Wealth'])}): {agent_data['Previous_Wealth'].tolist()}", flush=True)
                    if 'Savings' in agent_data.columns: print(f"    Problematic Savings ({type(agent_data['Savings'])}): {agent_data['Savings'].tolist()}", flush=True)
                    if 'Consumption' in agent_data.columns: print(f"    Problematic Consumption ({type(agent_data['Consumption'])}): {agent_data['Consumption'].tolist()}", flush=True)
                    if 'Utility' in agent_data.columns: print(f"    Problematic Utility ({type(agent_data['Utility'])}): {agent_data['Utility'].tolist()}", flush=True)
                    # Continue to the next profile
                    continue

            # else: # Optional: print message if data is missing for a profile on this plot
                 # print(f"  - No data found for profile '{profile_name}' at R={interest_rate:.2f}. Skipping.")


        # --- Finalize and Save TIME SERIES Figure for this interest rate ---
        if plot_successful:
            # Add legends to each subplot where at least one profile was plotted
            for i in range(len(axes)): # Iterate through all axes
                 if plotted_profiles_on_subplot[i]: # Check if any profile was successfully plotted on this subplot
                     axes[i].legend(fontsize='x-small', loc='best') # Adjust legend props if needed

            plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent title overlap (top slightly lower for suptitle)

            # Save the combined figure to the specified directory
            plot_filename_ts = os.path.join(output_dir_plots, f"Combined_TimeSeries_R{interest_rate:.2f}.png")
            try:
                plt.savefig(plot_filename_ts)
                print(f"  Saved combined time series plot: {plot_filename_ts}")
            except Exception as e:
                print(f"Error saving plot {plot_filename_ts}: {e}")
            plt.close(fig) # Close the time series figure for this interest rate
        else:
            print(f"  Skipping combined time series plot for R={interest_rate:.2f} (no successful simulations found or plotted).")
            plt.close(fig) # Close the empty figure

        if summed_savings_for_bar: # Check if we collected any data for the bar chart
            print(f"  Generating total savings bar chart for R = {interest_rate:.2f}")
            plt.figure(figsize=(10, 6)) # New figure for the bar chart

            # Get colors corresponding to the profiles plotted
            bar_colors = [profile_colors[p_name] for p_name in profile_names_for_bar]

            plt.bar(profile_names_for_bar, summed_savings_for_bar, color=bar_colors)

            plt.title(f"Total Savings Summed Over Time (R = {interest_rate:.2f})")
            plt.xlabel("Agent Profile")
            plt.ylabel("Total Savings ($)")
            plt.grid(True, axis='y', linestyle=':', alpha=0.7)
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # Use scientific notation if values large
            plt.xticks(rotation=45, ha='right') # Rotate labels if they overlap

            plt.tight_layout()

            # Save the bar chart figure
            plot_filename_bar = os.path.join(output_dir_plots, f"TotalSavings_Bar_R{interest_rate:.2f}.png")
            try:
                plt.savefig(plot_filename_bar)
                print(f"  Saved total savings bar chart: {plot_filename_bar}")
            except Exception as e:
                print(f"Error saving bar chart {plot_filename_bar}: {e}")
            plt.close() # Close the bar chart figure
        else:
            print(f"  Skipping total savings bar chart for R={interest_rate:.2f} (no data collected).")
        # ********************************************************************


    print("\n" + "="*70)
    print("Plot Generation Complete", flush=True)
    print("="*70)

# --- Restore Standard Output ---
sys.stdout = original_stdout # Reset stdout to console
print(f"\nSimulation stdout log saved to: {output_file_path}")
print(f"Plots saved to directory: {output_dir_plots}")
'''