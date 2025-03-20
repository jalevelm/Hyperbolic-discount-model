import sys 
import os 
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector  
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize  
from joblib import Parallel, delayed


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
    def __init__(self, unique_id, model, beta, delta, init_wealth, sigma, borrowing_limit):
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
        
        # --- Derived Parameters ---
        self.R_star = 1 + (1 - self.delta) / (self.beta * self.delta)   # Threshold interest rate
        
        # --- Value and Policy Function (for Markov Equilibrium) ---
        self.value_function_calculated = False  # Flag: Value iteration performed?
        self.V = None  # Value function (array, calculated later)
        self.g = None  # Policy function (array, calculated later)

        # --- Internal Tracking ---
        self.step_count = 0     # Number of steps taken

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
            print(f"Agent {self.unique_id}: First Step - Beta: {self.beta}, Delta: {self.delta}, R_star: {self.R_star}")
        
        # --- Debugging Prints (Start of Step) ---
        print(f"Agent {self.unique_id}: Step start. Wealth: {self.wealth}, Previous Savings: {self.previous_savings}")
        print(f"Agent {self.unique_id}: R = {self.model.interest_rate}, R_star = {self.R_star}")

        wealth_grid = self.model.wealth_grid    # Access the pre-defined wealth grid

        # --- Value Function and Policy Function Calculation (One-Time) ---
        if not self.value_function_calculated:
            print(f"Agent {self.unique_id}: Starting value iteration...")
            # Initialize value and policy functions as NumPy arrays
            V = np.zeros_like(wealth_grid)
            g = np.zeros_like(wealth_grid)

            # --- Value Iteration Algorithm ---
            iterations = 2000  # Maximum number of iterations
            tolerance = 1e-10  # Convergence tolerance
            iteration_count = 0   # Iteration counter

            for _ in range(iterations):
                iteration_count += 1 
                V_old = V.copy()    # Store the previous iteration's value function

                # Iterate over all possible wealth levels in the grid
                for i, k_val in enumerate(wealth_grid):
                    # Find the optimal savings and continuation value using the
                    # current value function (V) and the agent's parameters.
                    continuation_value, next_k, _ = self.optimize_savings(k_val, V, wealth_grid)

                    # Calculate consumption based on the budget constraint
                    consumption = self.model.interest_rate * k_val - next_k
                    consumption = max(consumption, 1e-9)    # Ensure consumption >= 0

                    # Update the value function: current utility + discounted future utility
                    V[i] = self.utility(consumption) + self.beta * continuation_value
                    g[i] = next_k # Store the optimal next-period wealth (savings)

                # --- Convergence Check ---
                if np.any(np.isnan(V)):   # Check for numerical instability
                    print(f"Agent {self.unique_id}: NaN detected in V after iteration {iteration_count}!")
                    break

                if np.any(np.abs(V) > 1e6): # Check for divergence
                    print(f"Agent {self.unique_id}: Value function is likely diverging at iteration {iteration_count}!")
                    raise ValueError("Value function is likely diverging")

                if np.max(np.abs(V - V_old)) < tolerance:   # Check for convergence
                    print(f"Agent {self.unique_id}: Value function converged after {iteration_count} iterations.")
                    break    
            else:
                # Executed if the loop completes without breaking (no convergence)
                print(f"Agent {self.unique_id}: Value function DID NOT converge after {iteration_count} iterations.")

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

        
        '''
        # --- R* Check and Savings Adjustment  ---
        # The following block attempts to enforce the theoretical saving/dissaving
        # behavior based on the comparison of the interest rate (R) and the
        # threshold interest rate (R*). 
        if self.model.interest_rate > self.R_star:
            # Agent SHOULD be saving
            self.savings = max(self.savings, self.previous_wealth * (1 + 0.001)) # Force a *very small* increase
        elif self.model.interest_rate < self.R_star:
            # Agent SHOULD be dissaving to the borrowing limit
            self.savings = self.borrowing_limit
        elif self.model.interest_rate == self.R_star:
            # Agent SHOULD hold wealth constant
            self.savings = self.previous_wealth
        '''

        # Calculate consumption based on the budget constraint
        consumption = self.model.interest_rate * self.wealth - self.savings
        consumption = max(consumption, 1e-9)  # Ensure positive consumption


        # --- Debugging Prints (End of Step) ---
        print(f"Agent {self.unique_id}: Step {self.step_count}")
        print(f"  Previous Wealth: {self.previous_wealth:.4f}")
        print(f"  Optimal Savings (before clipping): {optimal_savings:.4f}")  
        print(f"  Calculated Savings (after clipping): {self.savings:.4f}")
        print(f"  New Wealth: {self.wealth:.4f}")
        print(f"  Consumption: {consumption:.4f}")

        # --- State Updates ---
        self.previous_wealth = self.wealth  # Store current wealth for next step
        self.wealth = self.savings  # Next period's wealth is this period's savings
        self.previous_savings = self.savings    # Store current savings for next step
        print(f"Agent {self.unique_id}: Step end.  Wealth: {self.wealth:.2f}, Savings: {self.savings:.2f}, Consumption: {consumption:.2f}")
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

    
    def optimize_savings(self, k, V, wealth_grid):
        """
        Optimizes the agent's savings decision for a given level of wealth.

        This function finds the optimal next-period wealth (k_next) that
        maximizes the agent's present-biased discounted utility, given its
        current wealth (k), the value function (V), and the wealth grid.
        This implements the agent's optimality condition.

        Args:
            k (float): The agent's current wealth.
            V (np.ndarray): The value function, representing the maximized
                discounted lifetime utility attainable from each wealth level.
            wealth_grid (np.ndarray): The discretized grid of possible wealth
                levels.

        Returns:
            tuple: A tuple containing:
                - continuation_value (float): The discounted value of being at
                  the optimal next-period wealth level.
                - optimal_k_next (float): The optimal next-period wealth (savings).
                - final_total_utility (float): The total utility achieved with
                  the optimal savings choice.  (Used primarily for debugging.)

                Returns (0, k, 0) if the optimization fails.
        """
        beta = self.beta    # Present bias parameter
        delta = self.delta  # Discount factor
        R = self.model.interest_rate   # Gross interest rate

        def objective(k_next):
            """
            The objective function to be *minimized*.  Represents the
            negative of the agent's present-biased utility.

            Args:
                k_next (float): The next period's wealth (agent's choice variable).

            Returns:
                float: The negative of the total utility (current utility +
                    discounted future utility).
            """
            # Calculate consumption based on the budget constraint.
            consumption = R * k - k_next
            if consumption <= 1e-6:
                # If consumption is very small (or negative), return a very
                # large negative utility (represented by positive infinity).
                # This effectively penalizes infeasible or extremely low
                # consumption choices.
                return np.inf

            # Ensure k_next stays within the bounds of the wealth grid.
            future_wealth = k_next
            future_wealth = np.clip(future_wealth, wealth_grid.min(), wealth_grid.max()) # Clip here.

            # Interpolate to find the value of being at future_wealth,
            # given the value function V.  This approximates V(future_wealth).
            future_utility = delta * np.interp(future_wealth, wealth_grid, V)

            # Calculate the total utility: current utility from consumption +
            # present-biased discounted future utility.
            total_utility = self.utility(consumption) + beta * future_utility
            
            # Return the *negative* of total utility, because we're using
            # a minimization routine.
            return -total_utility

        # Define the lower bound for the optimization.  The agent cannot
        # save less than the borrowing limit (or zero, if borrowing is not allowed).
        lower_bound = max(self.borrowing_limit, 0)

        # Use scipy.optimize.minimize_scalar to find the value of k_next
        # that minimizes the objective function (maximizes utility).
        # We use the 'bounded' method to constrain the search within the
        # feasible range (lower_bound to R*k).
        result = optimize.minimize_scalar(objective, bounds=(lower_bound, R * k), method='bounded')


        if result.success:
            # If the optimization was successful:
            optimal_k_next = result.x   # Extract the optimal k_next.

            # Ensure the optimal value is within grid bounds (should already be,
            # but this is a safeguard).
            optimal_k_next = np.clip(optimal_k_next, wealth_grid.min(), wealth_grid.max())

            # Calculate the continuation value (discounted future utility)
            # at the optimal next-period wealth.
            continuation_value = delta * np.interp(optimal_k_next, wealth_grid, V)

            # Calculate consumption based on the optimal savings choice.
            final_consumption = R * k - optimal_k_next

            # Calculate total utility (for debugging/verification).
            final_total_utility = self.utility(final_consumption) + beta * continuation_value

            return continuation_value, optimal_k_next, final_total_utility
        else:
            # If the optimization failed, print an error message and return
            # default values.
            print(f"Agent {self.unique_id}: Optimization FAILED for k={k}")
            print(result)   # Print the optimization result for debugging
            return 0, k, 0  # Return default values




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
    
    def __init__(self, agent_profile, interest_rate, sigma, wealth_dist):
        """
        Initializes the SavingModel.

        Args:
            agent_profile (dict): A dictionary containing the parameters
                ('beta' and 'delta') for the agent's hyperbolic discounting
                preferences.
            interest_rate (float): The constant gross interest rate.
            sigma (float): The coefficient of relative risk aversion.
            wealth_dist (list): The distribution of initial wealth.
        """

        super().__init__()  # Initialize the Model superclass

        # --- Model Parameters ---
        self.num_agents = 1  #Number of agents
        self.interest_rate = interest_rate  # Constant gross interest rate
        self.sigma = sigma  # inv. of intertemporal substitution
        self.max_wealth = 50000000  # Upper bound for the wealth grid
        self.borrowing_limit = 0    # Lower bound for wealth (no borrowing)
        self.wealth_dist = wealth_dist  # Initial wealth distribution

        # --- Mesa Components ---
        # Grid is not strictly necessary for a single agent but is kept for
        # compatibility with potential future multi-agent extensions.
        self.grid = MultiGrid(10, 10, True)  # A 10x10 grid
        self.schedule = RandomActivation(self)  # Random activation scheduler

        # --- Create the Wealth Grid ---
        # A discrete set of wealth levels used for value function iteration.
        self.wealth_grid = np.linspace(1e-6, self.max_wealth, 1000) 


        # --- Create the Agent ---
        self.create_agent(agent_profile)    # Create and add the agent

        # --- Data Collection ---
        self.datacollector = DataCollector(
            agent_reporters={
                "Wealth": "wealth",
                "Savings": "savings",
                # Calculate consumption using the budget constraint.
                "Consumption": lambda a: a.model.interest_rate * a.previous_wealth - a.savings,
                "R_star": "R_star",
                "Interest_Rate": lambda a: a.model.interest_rate,
                # Calculate utility from consumption.
                "Utility": lambda a: a.utility(a.model.interest_rate * a.previous_wealth - a.savings),
            }
        )
        print("Model initialized")

    def create_agent(self, profile):
        """
        Creates a single SavingAgent and adds it to the model.

        Args:
            profile (dict): A dictionary containing the agent's 'beta' and
                'delta' values, defining its hyperbolic discounting preferences.
        """
        print(f"Creating agent with profile: {profile}")

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

        # --- Create and Add the Agent ---
        # Create a SavingAgent instance with the specified parameters.
        # Note: The wealth grid is now available as self.wealth_grid.
        agent = SavingAgent(0, self, profile["beta"], profile["delta"], init_wealth, self.sigma, self.borrowing_limit)
        self.schedule.add(agent)  # Add the agent to the scheduler
        self.grid.place_agent(agent, (0, 0))  # Place agent on grid (position irrelevant)
        print(f"Agent created and added to schedule. Initial wealth: {init_wealth}")
       


    def step(self):
        """
        Advances the model by one time step.

        This involves:
        1. Collecting data from the current state.
        2. Advancing the agent (which performs its saving/consumption decision).
        """
        print("Model step start")  # Debug print
        self.datacollector.collect(self)    # Collect data
        self.schedule.step()    # Advance the agent (and scheduler)
        print("Model step end")  # Debug print

# ---------------------------------------------------------- Model run ------------------------------------------------------------------------

# --- Define Agent Profiles ---
agent_profiles = {
    "impulsive": {"beta": 0.6, "delta": 0.85},  # lower beta = more present bias, lower delta = less patient
    "planner": {"beta": 0.8, "delta": 0.98},  # Higher beta = less present bias, higer delta = more patient
    "procrastinator": {"beta": 0.6, "delta": 0.98},  # low beta = more present bias, high delta = more patient, values also future consumption
    "moderate": {"beta": 0.7, "delta": 0.96},  # Base values
}

sigma = 0.4387 # elasticity of satisfaction

wealth_dist = [
    (0.41520521, (0, 9999)),  # Less than 10,000 USD
    (0.477205824, (10000, 99999)),  # Between 10,000 and 100,000 USD
    (0.103122194, (100000, 999999)),  # Between 100,000 and 1,000,000 USD
    (0.004466772, (1000000, 10000000))  # More than 1,000,000 USD 
]

# --- Redirect Output to File ---
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
output_file_path = os.path.join(output_dir, "simulation_output.txt")

with open(output_file_path, "w") as output_file:
    sys.stdout = output_file  # Redirect standard output to the file

# -------------------------------------------------------- Run Simulation-----------------------
    for profile_name, profile in agent_profiles.items():
        print(f"\n                                                                                    ----- Running Simulations for {profile_name} -----")

        # Test different interest rates
        for interest_rate in [1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.10, 1.20, 1.30]:  # Example interest rates
            print(f"\n--- Interest Rate: {interest_rate} ---")
            model = SavingModel(profile, interest_rate, sigma, wealth_dist)
            for i in range(5):  # Number of steps
                model.step()

            # Analyze data and plot
            agent = model.schedule.agents[0]
            if agent.value_function_calculated:     # Check if value iteration ran
                agent_data = model.datacollector.get_agent_vars_dataframe()
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.plot(model.wealth_grid, agent.V)
                plt.title(f"Agent {agent.unique_id}: Value Function ({profile_name}, R={interest_rate})")
                plt.xlabel("Wealth (k)")
                plt.ylabel("V(k)")

                plt.subplot(1, 2, 2)
                plt.plot(model.wealth_grid, agent.g)
                plt.plot(model.wealth_grid, model.wealth_grid, color='red', linestyle='--', label="45-degree line")
                plt.title(f"Agent {agent.unique_id}: Policy Function ({profile_name}, R={interest_rate})")
                plt.xlabel("Current Wealth (k)")
                plt.ylabel("Next Period Wealth (g(k))")
                plt.legend()
                plot_filename = f"output/{profile_name}_R{interest_rate}_value_policy_plots.png" #Saving plots
                plt.savefig(plot_filename)
                plt.close()


            plt.figure(figsize=(12, 8))

            plt.subplot(2, 2, 1)
            plt.plot(agent_data["Wealth"].values, label="Wealth")
            plt.plot(agent_data["Savings"].values, label="Savings")
            plt.xlabel("Time")
            plt.ylabel("Amount")
            plt.title(f"{profile_name} - R={interest_rate} - Wealth & Savings") # Add to title
            plt.legend()

            plt.subplot(2, 2, 2)
            plt.plot(agent_data["Consumption"].values, label="Consumption")
            plt.xlabel("Time")
            plt.ylabel("Consumption")
            plt.title(f"{profile_name} - R={interest_rate} - Consumption")  # Add to title
            plt.legend()

            plt.subplot(2, 2, 3)
            plt.plot(agent_data["Utility"].values, label="Utility")
            plt.xlabel("Time")
            plt.ylabel("Utility")
            plt.title(f"{profile_name} - R={interest_rate} - Utility")  # Add to title
            plt.legend()

            plt.subplot(2, 2, 4)
            plt.plot(agent_data["Wealth"].values, agent_data["Savings"].values, label="Wealth vs Savings")
            plt.xlabel("Wealth")
            plt.ylabel("Savings")
            plt.title(f"{profile_name} - R={interest_rate} - Wealth vs Savings")  # Add to title
            plt.legend()

            plt.tight_layout()
            # Save the figure to a file
            plot_filename = f"output/{profile_name}_R{interest_rate}_plots.png"
            plt.savefig(plot_filename)
            plt.close()  # Close the figure to free up memory

sys.stdout = sys.__stdout__ #resets to printing to console

print(f"Simulation output saved to: {output_file_path}")