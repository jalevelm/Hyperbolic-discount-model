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
    An agent with hyperbolic discounting preferences, making saving decisions
    """
    def __init__(self, unique_id, model, beta, delta, init_wealth, sigma, borrowing_limit):
        super().__init__(unique_id, model)
        self.beta = beta
        self.delta = delta
        self.wealth = init_wealth
        self.sigma = sigma
        self.savings = 0
        self.R_star = 1 + (1 - self.delta) / (self.beta * self.delta)
        self.borrowing_limit = borrowing_limit
        self.previous_savings = None
        self.init_wealth = init_wealth
        self.step_count = 0
        self.previous_wealth = init_wealth 
        self.value_function_calculated = False # Flag for value iteration
        self.V = None  # Store the value function
        self.g = None  # Store the policy function

    def step(self):
        if self.step_count == 0:
            print(f"Agent {self.unique_id}: First Step - Beta: {self.beta}, Delta: {self.delta}, R_star: {self.R_star}")
        
        print(f"Agent {self.unique_id}: Step start. Wealth: {self.wealth}, Previous Savings: {self.previous_savings}")
        print(f"Agent {self.unique_id}: R = {self.model.interest_rate}, R_star = {self.R_star}")

        wealth_grid = self.model.wealth_grid

        # Value Iteration (only once, at the beginning)
        if not self.value_function_calculated:
            print(f"Agent {self.unique_id}: Starting value iteration...")
            V = np.zeros_like(wealth_grid)
            g = np.zeros_like(wealth_grid)
            iterations = 1000  # Increased iterations
            tolerance = 1e-6  # Tighter tolerance
            iteration_count = 0

            for _ in range(iterations):
                iteration_count += 1 
                V_old = V.copy()
                for i, k_val in enumerate(wealth_grid):
                    continuation_value, next_k, _ = self.optimize_savings(k_val, V, wealth_grid)
                    consumption = self.model.interest_rate * k_val - next_k
                    consumption = max(consumption, 1e-9)
                    V[i] = self.utility(consumption) + self.beta * continuation_value
                    g[i] = next_k

                if np.any(np.isnan(V)):
                    print(f"Agent {self.unique_id}: NaN detected in V after iteration {iteration_count}!")
                    break

                if np.any(np.abs(V) > 1e6):
                    print(f"Agent {self.unique_id}: Value function is likely diverging at iteration {iteration_count}!")
                    raise ValueError("Value function is likely diverging!")

                if np.max(np.abs(V - V_old)) < tolerance:
                    print(f"Agent {self.unique_id}: Value function converged after {iteration_count} iterations.")
                    break    
            else:
                print(f"Agent {self.unique_id}: Value function DID NOT converge after {iteration_count} iterations.")

            self.V = V
            self.g = g
            self.value_function_calculated = True

        # Agent's Decision 
        self.savings = np.interp(self.wealth, wealth_grid, self.g) 
        self.savings = np.clip(self.savings, self.borrowing_limit, self.model.interest_rate * self.wealth)
        consumption = self.model.interest_rate * self.wealth - self.savings
        consumption = max(consumption, 1e-9)  # Ensure positive consumption

        #--R* check---
        if self.model.interest_rate > self.R_star:
            # Agent SHOULD be saving (or at worst, be indifferent at k_t+1 = k_t)
            self.savings = max(self.savings, self.previous_wealth)  # Ensure k_t+1 >= k_t

        elif self.model.interest_rate < self.R_star:
            # Agent SHOULD be dissaving to the borrowing limit
            self.savings = self.borrowing_limit 


        # --- Debugging Prints (Once per Step) ---
        print(f"Agent {self.unique_id}: Step {self.step_count}")
        print(f"  Previous Wealth: {self.previous_wealth:.4f}")
        print(f"  Calculated Savings: {self.savings:.4f}")
        print(f"  New Wealth: {self.wealth:.4f}")
        print(f"  Consumption: {consumption:.4f}")
        # --- End Debugging Prints ---

        # Update wealth and previous_wealth *CORRECTLY*
        self.previous_wealth = self.wealth  # *Before* updating wealth
        self.wealth = self.savings # Wealth in next period is savings from this period.
        self.previous_savings = self.savings
        print(f"Agent {self.unique_id}: Step end.  Wealth: {self.wealth:.2f}, Savings: {self.savings:.2f}, Consumption: {consumption:.2f}")
        self.step_count += 1

    def utility(self, consumption):
        """
        Calculate utility for a given consumption level.
        """
        epsilon = 1e-9  # Small positive value
        consumption = max(consumption, epsilon)  # Ensure consumption is never too small
        if self.sigma == 1:
            return np.log(consumption)
        return consumption**(1 - self.sigma) / (1 - self.sigma)

    
    def optimize_savings(self, k, V, wealth_grid):
        beta = self.beta
        delta = self.delta
        R = self.model.interest_rate

        def objective(k_next):
            consumption = R * k - k_next
            if consumption <= 1e-6:
                return np.inf

            future_wealth = k_next
            future_wealth = np.clip(future_wealth, wealth_grid.min(), wealth_grid.max()) # Clip here.

            future_utility = delta * np.interp(future_wealth, wealth_grid, V)
            total_utility = self.utility(consumption) + beta * future_utility
            return -total_utility

        lower_bound = max(self.borrowing_limit, 0)
        result = optimize.minimize_scalar(objective, bounds=(lower_bound, R * k), method='bounded')


        if result.success:
            optimal_k_next = result.x
            optimal_k_next = np.clip(optimal_k_next, wealth_grid.min(), wealth_grid.max()) # Ensure bounds

            continuation_value = delta * np.interp(optimal_k_next, wealth_grid, V)
            final_consumption = R * k - optimal_k_next
            final_total_utility = self.utility(final_consumption) + beta * continuation_value

            return continuation_value, optimal_k_next, final_total_utility
        else:
            print(f"Agent {self.unique_id}: Optimization FAILED for k={k}")
            print(result)
            return 0, k, 0




class SavingModel(Model):
 
    def __init__(self, agent_profile, interest_rate, sigma, wealth_dist):

        super().__init__()  # Initialize the Model superclass
        self.num_agents = 1 #number of agents
        self.grid = MultiGrid(10, 10, True)  # Torus grid, not used but kept for compatibility
        self.schedule = RandomActivation(self)
        self.interest_rate = interest_rate
        self.sigma = sigma
        self.max_wealth = 50000000  # Adjusted for faster convergence
        self.borrowing_limit = 0
        self.wealth_dist = wealth_dist
        self.create_agent(agent_profile)
        self.wealth_grid = np.linspace(1e-6, self.max_wealth, 500) # Increased density, adjusted range


        # Data collection
        self.datacollector = DataCollector(
            agent_reporters={
                "Wealth": "wealth",
                "Savings": "savings",
                "Consumption": lambda a: a.model.interest_rate * a.previous_wealth - a.savings,
                "R_star": "R_star",
                "Interest_Rate": lambda a: a.model.interest_rate,
                "Utility": lambda a: a.utility(a.model.interest_rate * a.previous_wealth - a.savings),
            }
        )
        print("Model initialized")

    def create_agent(self, profile):
            print(f"Creating agent with profile: {profile}")
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

            agent = SavingAgent(0, self, profile["beta"], profile["delta"], init_wealth, self.sigma, self.borrowing_limit)
            self.schedule.add(agent)
            self.grid.place_agent(agent, (0, 0))  # Position irrelevant
            print(f"Agent created and added to schedule. Initial wealth: {init_wealth}")        


    def step(self):
        """
        Advance the model by one time step.
        """
        print("Model step start")  # Debug print
        self.datacollector.collect(self)
        self.schedule.step()
        print("Model step end")  # Debug print

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
            for i in range(10):  # Number of steps
                model.step()

            # Analyze data
            agent_data = model.datacollector.get_agent_vars_dataframe()

    # --- Plot individual agent data ---
            agent = model.schedule.agents[0]  # Access the agent
            if agent.value_function_calculated:
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