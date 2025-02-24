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

    def step(self):
        """
        Define the agent's actions in each time step.
        """

        # Discretize possible wealth levels
        wealth_grid = np.linspace(0, self.model.max_wealth, 100)  # Adjust grid size as needed


         # Print global R and agent's R_star
        print(f"Agent {self.unique_id}: Global R = {self.model.interest_rate}, R* = {self.R_star}")


        # Initialize value function and saving function
        V = np.zeros_like(wealth_grid)
        g = np.zeros_like(wealth_grid)

        # Value iteration and policy improvement
        for _ in range(50):  # Adjust number of iterations as needed
            V_old = V.copy()

            # Parallelize the loop over wealth_grid
            results = Parallel(n_jobs=-1)(delayed(self.optimize_savings)(k, V_old, self.model.max_wealth, wealth_grid) for k in wealth_grid)
            V = np.array([result[0] for result in results])
            g = np.array([result[1] for result in results])

        # Find optimal savings for current wealth
        self.savings = g[np.argmin(np.abs(wealth_grid - self.wealth))]

        # --- Print detailed agent data ---
        print(f"Agent {self.unique_id}: Wealth = {self.wealth:.2f}, Savings = {self.savings:.2f}, "
            f"Consumption = {self.wealth - self.savings:.2f}, R_star = {self.R_star:.2f}, "
            f"Interest Rate = {self.model.interest_rate:.2f}, "
            f"Utility = {self.utility(self.wealth - self.savings):.2f}")

        # Update wealth
        self.wealth -= self.savings

        # Find optimal savings for current wealth
        self.savings = g[np.argmin(np.abs(wealth_grid - self.wealth))]



    def utility(self, consumption):
        """
        Calculate utility for a given consumption level.
        """
        if self.sigma == 1:
            return np.log(consumption)
        else:
            return consumption**(1 - self.sigma) / (1 - self.sigma)
    
    def optimize_savings(self, k, V_old, max_wealth, wealth_grid):
        """
        Calculate the optimal savings amount.

        Args:
        k: Current wealth level.
        V_old: Value function from the previous iteration.
        max_wealth: Maximum wealth level in the model.
        wealth_grid: Discretized wealth grid.

        Returns:
        tuple: (maximum value, corresponding saving amount)
        """
        # Adjust the range of possible savings based on the relationship 
        # between the model's interest rate and the agent's R_star
        if self.model.interest_rate > self.R_star:
            # Agent has a natural inclination to save
            possible_savings = np.linspace(0, k, 100)  # Adjust grid size as needed
        else:
            # Agent has a natural inclination to dis-save
            possible_savings = np.linspace(-k, k, 100)  # Adjust grid size as needed

        # Enforce borrowing limit  
        possible_savings = possible_savings[k - possible_savings >= self.borrowing_limit] 

        #  Implement No Reversal Principle 
        if hasattr(self, 'previous_savings'):
            if self.model.interest_rate > self.R_star and self.previous_savings <= 0:
                possible_savings = possible_savings[possible_savings <= 1e-6]
            elif self.model.interest_rate < self.R_star and self.previous_savings >= 0:
                possible_savings = possible_savings[possible_savings >= -1e-6]

        # Vectorized calculation of V_next
        consumption = k - possible_savings  # Calculate consumption for all possible savings
         # --- INTERPOLATION  ---
        # Calculate the value of the next period's value function (V_old)
        # using linear interpolation.  Instead of just taking the value
        # at the nearest grid point, the value is estimated between grid
        # points. This makes the value function approximation much more
        # accurate.
        interpolated_V_old = np.interp(possible_savings, wealth_grid, V_old)
        V_next = self.utility(consumption) + self.delta * interpolated_V_old
        # --- END OF INTERPOLATION ---

        max_index = np.argmax(V_next)
        best_savings = possible_savings[max_index]

        self.previous_savings = best_savings  # Update previous_savings

        return np.max(V_next), best_savings



class SavingModel(Model):
    """
    A model with multiple saving agents.
    """
    def __init__(self, N, width, height, interest_rate, sigma, beta_ranges, delta_ranges, wealth_dist):
        """
        Initialize the model.

        Argumentss:
            N: Number of agents in the model.
            width: Width of the grid.
            height: Height of the grid.
            interest_rate: The global interest rate in the economy.
            sigma: satisfaction elasticity (how much does satisfaction changes when consumption changes) (same for all agents in this version).
            beta_ranges: Tuple (min, max) for the range of beta values for agents.
            delta_ranges: Tuple (min, max) for the range of delta values for agents.
            wealth_dist: A list of tuples defining the initial wealth distribution.
                         Each tuple is (probability, (min_wealth, max_wealth)).
        """
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)  # Torus grid
        self.schedule = RandomActivation(self)
        self.interest_rate = interest_rate
        self.sigma = sigma
        self.max_wealth = 10000000  # Adjust as needed
        self.borrowing_limit = 0 

        # Create agents
        for i in range(self.num_agents):
            # Sample beta and delta from ranges
            beta = random.choice(np.arange(beta_ranges[0], beta_ranges[1] + 0.01, 0.01)) 
            delta = random.choice(np.arange(delta_ranges[0], delta_ranges[1] + 0.01, 0.01)) 

            # Sample initial wealth based on distribution
            rand_num = random.random()
            cumulative_prob = 0
            for prob, wealth_range in wealth_dist:  
                cumulative_prob += prob
                if rand_num <= cumulative_prob:
                    init_wealth = random.randint(wealth_range[0], wealth_range[1])
                    break

            # Create the agent with the chosen parameters
            a = SavingAgent(i, self, beta, delta, init_wealth, sigma, self.borrowing_limit)
            self.schedule.add(a)

             # Place the agent at a random location on the grid
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        # Data collection
        self.datacollector = DataCollector(
            agent_reporters={
                "Wealth": "wealth",
                "Savings": "savings",
                "Consumption": lambda a: a.wealth - a.savings,
                "R_star": "R_star",
                "Interest_Rate": lambda a: a.model.interest_rate,
                "Utility": lambda a: a.utility(a.wealth - a.savings)
            }


        
        )

    def step(self):
        """
        Advance the model by one time step.
        """
        self.datacollector.collect(self)
        self.schedule.step()

#_____________________________________________________________________ Example usage_______________________________________________________________________________
N = 1  # Number of agents
width = 10
height = 10
interest_rate = 0.15  # Annual interest rate
sigma = 0.4387 # elasticity of satisfaction

# Example usage with ranges for beta and delta
beta_ranges = (0.70, 1)  # Beta range 
delta_ranges = (0.95, 1)  # Delta range 

# Initial wealth distribution 
wealth_dist = [
    (0.41520521, (0, 9999)),  # Less than 10,000 USD
    (0.477205824, (10000, 99999)),  # Between 10,000 and 100,000 USD
    (0.103122194, (100000, 999999)),  # Between 100,000 and 1,000,000 USD
    (0.004466772, (1000000, 10000000))  # More than 1,000,000 USD 
]


# Create a model instance with the specified parameters
model = SavingModel(N, width, height, interest_rate, sigma, beta_ranges, delta_ranges, wealth_dist)

# Run the model
for i in range(24):  # Run for X months 
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