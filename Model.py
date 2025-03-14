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
        """
        Initialize a new agent.

        Argumentss:
            unique_id: Unique identifier for the agent.
            model: The model the agent belongs to.
            beta: Present bias parameter (how much the agent discounts the future, A lower β implies a stronger preference for present consumption.this makes individuals discount the next period more heavily than any two future periods).
            delta: Standard discount factor (general impatience).
            init_wealth: Initial wealth of the agent.
            salary: Agent's regular income.
            sigma: Satisfaction elasticity: how much an agent's satisfaction changes when its consumption changes.
        """
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
        self.previous_wealth = 0
        print(f"Agent {unique_id} initialized. init_wealth: {self.init_wealth}, wealth: {self.wealth}")  # Debug print

    def step(self):
        """
        Define the agent's actions in each time step.
        """
        if self.step_count == 0:
            print(f"Agent {self.unique_id}: First Step - Beta: {self.beta}, Delta: {self.delta}, R_star: {self.R_star}") # Debug print
        self.step_count += 1

        print(f"Agent {self.unique_id}: Step start. Wealth: {self.wealth}, Previous Savings: {self.previous_savings}") # Debug print

        self.previous_wealth = self.wealth

        wealth_grid = self.model.wealth_grid
        k = self.wealth
        V = np.zeros_like(wealth_grid)
        g = np.zeros_like(wealth_grid)
        V_old = np.zeros_like(wealth_grid)  # Initialize V_old to zeros
        print(f"Agent {self.unique_id}: Starting value iteration...")
        iterations = 200
        
        for _ in range(iterations):
            for i, k_val in enumerate(wealth_grid):
                continuation_value, next_k = self.optimize_savings(k_val, V_old, wealth_grid)
                # *** Safeguard for very small consumption ***
                consumption = self.model.interest_rate * k_val - next_k
                epsilon = 1e-9
                consumption = max(consumption, epsilon) #Ensure consumption is not too small
                V[i] = self.utility(consumption) + self.beta * continuation_value # Pass the corrected value
                g[i] = next_k

            if np.any(np.isnan(V)):  # *** ADD THIS CHECK ***
                print(f"Agent {self.unique_id}: NaN detected in V after iteration {_}!")
                # You might even want to *stop* the simulation here with 'break'
                break    

            if np.max(np.abs(V - V_old)) < 1e-3:
                print(f"Agent {self.unique_id}: Value function converged after {_} iterations.")
                break
            V_old = V.copy()
        else:
            print(f"Agent {self.unique_id}: Value function DID NOT converge after {iterations} iterations.")

        closest_index = np.argmin(np.abs(wealth_grid - self.wealth))
        self.savings = g[closest_index]
        self.wealth = self.model.interest_rate * self.previous_wealth - (self.previous_wealth - self.savings)
        consumption = self.model.interest_rate * self.previous_wealth - self.savings
        self.previous_savings = self.savings
        print(f"Agent {self.unique_id}: Step end.  Wealth: {self.wealth:.2f}, Savings: {self.savings:.2f}, Consumption: {consumption:.2f}")


    def print_optimization_details(self, k, V_old, wealth_grid):
        print(f"\n--- Agent {self.unique_id}: Optimizing Savings for k = {k:.2f}, Initial Wealth: {self.init_wealth} ---")

        if self.model.interest_rate > self.R_star:
            possible_savings = np.linspace(0, k, 50)  # Reduced for brevity
        else:
            lower_bound = max(-abs(self.borrowing_limit), -k)  # Correct lower bound
            possible_savings = np.linspace(lower_bound, k, 50)


        if self.previous_savings is not None:
            print(f"  Previous Savings: {self.previous_savings:.2f}")
            if self.model.interest_rate > self.R_star and self.previous_savings <= 0:
                possible_savings = possible_savings[possible_savings <= 1e-6]
            elif self.model.interest_rate < self.R_star and self.previous_savings >= 0:
                possible_savings = possible_savings[possible_savings >= -1e-6]
        print(f"  Possible Savings (after no-reversal if applicable): {possible_savings}")

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
        """Optimize savings for a given wealth level k (simplified)."""
        beta = self.beta
        delta = self.delta
        R = self.model.interest_rate

        def objective(k_next):
            consumption = R * k - k_next
            if consumption <= 1e-6:
                return np.inf  # Avoid log(0) and enforce positive consumption

            future_wealth = k_next
            future_wealth = np.clip(future_wealth, wealth_grid.min(), wealth_grid.max())
            future_utility = delta * np.interp(future_wealth, wealth_grid, V, left=V[0], right=V[-1])
            total_utility = self.utility(consumption) + beta * future_utility
            return -total_utility

            if abs(k - self.wealth) < 1000:  # Adjust the tolerance (100) as needed
                print(f"    k_next: {k_next:.2f}, consumption: {consumption:.2f}, future_utility: {future_utility:.2f}, total_utility: {total_utility}")
            return -total_utility

        lower_bound = max(self.borrowing_limit, 0)  # Borrowing constraint
        result = optimize.minimize_scalar(objective, bounds=(lower_bound, R * k), method='bounded')

        if result.success:
            optimal_k_next = result.x
            optimal_k_next = np.clip(optimal_k_next, wealth_grid.min(), wealth_grid.max())
            continuation_value = delta * np.interp(optimal_k_next, wealth_grid, V, left=V[0], right=V[-1])
            return continuation_value, optimal_k_next
        else:
            print(f"Agent {self.unique_id}: Optimization FAILED for k={k}")
            return 0, k



class SavingModel(Model):
 
    def __init__(self, agent_profile, interest_rate, sigma, wealth_dist):
  
        super().__init__()  # Initialize the Model superclass
        self.num_agents = 1 #number of agents
        self.grid = MultiGrid(10, 10, True)  # Torus grid
        self.schedule = RandomActivation(self)
        self.interest_rate = interest_rate
        self.sigma = sigma
        self.max_wealth = 10000000  # Adjust as needed
        self.borrowing_limit = 0 
        self.wealth_dist = wealth_dist
        self.create_agent(agent_profile)
        self.wealth_grid = np.linspace(0, self.max_wealth * self.interest_rate, 50) #ajust wealth grid size


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
        print(f"Creating agent with profile: {profile}")  # Debug print
        rand_num = random.random()
        cumulative_prob = 0
        init_wealth = 0  # Default initialization
        for prob, wealth_range in self.wealth_dist:
            cumulative_prob += prob
            if rand_num <= cumulative_prob:
                init_wealth = random.randint(wealth_range[0], wealth_range[1])
                break
        else:  
            init_wealth = random.randint(max(1, self.wealth_dist[-1][1][0]), self.wealth_dist[-1][1][1])
        
        # Create the agent with the specified profile's beta and delta
        agent = SavingAgent(0, self, profile["beta"], profile["delta"], init_wealth, self.sigma, self.borrowing_limit)
        self.schedule.add(agent)
        self.grid.place_agent(agent, (0, 0))  # Place agent (position irrelevant)
        print(f"Agent created and added to schedule. Initial wealth: {init_wealth}")  # Debug print

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


# -------------------------------------------------------- Run Simulation-----------------------
for profile_name, profile in agent_profiles.items():
    print(f"\n----- Running Simulations for {profile_name} -----")

    # Set a fixed interest rate.  You can experiment with different values.
    interest_rate = 1.03  # Example: 3% interest rate

    model = SavingModel(profile, interest_rate, sigma, wealth_dist)
    for i in range(10):  # Number of steps
        model.step()

    # Analyze data

    agent_data = model.datacollector.get_agent_vars_dataframe()

    # --- Plot individual agent data ---
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(agent_data["Wealth"].values, label="Wealth")
    plt.plot(agent_data["Savings"].values, label="Savings")
    plt.xlabel("Time")
    plt.ylabel("Amount")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(agent_data["Consumption"].values, label="Consumption")
    plt.xlabel("Time")
    plt.ylabel("Consumption")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(agent_data["Utility"].values, label="Utility")
    plt.xlabel("Time")
    plt.ylabel("Utility")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(agent_data["Wealth"].values, agent_data["Savings"].values, label="Wealth vs Savings")
    plt.xlabel("Wealth")
    plt.ylabel("Savings")
    plt.legend()

    plt.tight_layout()
    plt.show()
