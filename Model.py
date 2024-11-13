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
    def __init__(self, unique_id, model, beta, delta, init_wealth, sigma):
        """
        Initialize a new agent.

        Argumentss:
            unique_id: Unique identifier for the agent.
            model: The model the agent belongs to.
            beta: Present bias parameter (how much the agent discounts the future, A lower β implies a stronger preference for present consumption.this makes individuals discount the next period more heavily than any two future periods).
            delta: Standard discount factor (general impatience).
            init_wealth: Initial wealth of the agent.
            salary: Agent's regular income.
            sigma: Risk aversion parameter (how much the agent dislikes uncertainty).
        """
        super().__init__(unique_id, model)
        self.beta = beta
        self.delta = delta
        self.wealth = init_wealth  
        self.sigma = sigma
        self.savings = 0 

    def step(self):
        """
        Define the agent's actions in each time step.
        """

        # Discretize possible wealth levels
        wealth_grid = np.linspace(0, self.model.max_wealth, 10)  # Adjust grid size as needed

        # Initialize value function and saving function
        V = np.zeros_like(wealth_grid)
        g = np.zeros_like(wealth_grid)

        # Value iteration and policy improvement
        for _ in range(10):  # Adjust number of iterations as needed
            V_old = V.copy()

            # Parallelize the loop over wealth_grid
            results = Parallel(n_jobs=-1)(delayed(self.optimize_savings)(k, V_old, self.model.max_wealth, wealth_grid) for k in wealth_grid)
            V = np.array([result[0] for result in results])
            g = np.array([result[1] for result in results])

        # Find optimal savings for current wealth
        self.savings = g[np.argmin(np.abs(wealth_grid - self.wealth))]

        # Update wealth
        self.wealth -= self.savings

    def utility(self, consumption):
        """
        Calculate utility for a given consumption level.
        """
        if self.sigma == 1:
            return np.log(consumption)
        else:
            return consumption**(1 - self.sigma) / (1 - self.sigma)
    
    def optimize_savings(self, k, V_old, max_wealth, wealth_grid):
        possible_savings = np.linspace(0, k, 5)  # Adjust grid size as needed

        # Vectorized calculation of V_next
        consumption = k - possible_savings  # Calculate consumption for all possible savings
        V_next = self.utility(consumption) + self.delta * V_old[np.argmin(np.abs(wealth_grid - possible_savings[:, np.newaxis]), axis=1)]  # Calculate the value function for all possible savings

        return np.max(V_next), possible_savings[np.argmax(V_next)]  # Return the maximum value and the corresponding saving amount


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
            sigma: Risk aversion parameter (same for all agents in this version).
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

        # Create agents
        for i in range(self.num_agents):
            # Sample beta and delta from ranges
            beta = random.choice(np.arange(beta_ranges[0], beta_ranges[1] + 0.1, 0.1)) 
            delta = random.choice(np.arange(delta_ranges[0], delta_ranges[1] + 0.1, 0.1)) 

            # Sample initial wealth based on distribution
            rand_num = random.random()
            cumulative_prob = 0
            for prob, wealth_range in wealth_dist:  
                cumulative_prob += prob
                if rand_num <= cumulative_prob:
                    init_wealth = random.randint(wealth_range[0], wealth_range[1])
                    break

            # Create the agent with the chosen parameters
            a = SavingAgent(i, self, beta, delta, init_wealth, sigma)
            self.schedule.add(a)

             # Place the agent at a random location on the grid
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        # Data  collection
        self.datacollector = DataCollector(
            model_reporters={"Average_Wealth": lambda m: np.mean([a.wealth for a in m.schedule.agents]),
                             "Average_Savings": lambda m: np.mean([a.savings for a in m.schedule.agents])},  # Collect average savings
            agent_reporters={"Wealth": "wealth", "Savings": "savings"}  # Collect individual savings
        )

    def step(self):
        """
        Advance the model by one time step.
        """
        self.datacollector.collect(self)
        self.schedule.step()

# Example usage
N = 10  # Number of agents
width = 10
height = 10
interest_rate = 0.2  # Annual interest rate
sigma = 0.8 # Example risk aversion parameter

# Example usage with ranges for beta and delta
beta_ranges = (0.9, 1)  # Beta range from 0 to 1
delta_ranges = (0.9, 1)  # Delta range from 0 to 0.9

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
for i in range(12):  # Run for X months 
    model.step()

# Analyze data
agent_data = model.datacollector.get_agent_vars_dataframe()
model_data = model.datacollector.get_model_vars_dataframe()

# Plot average wealth over time
plt.plot(model_data["Average_Wealth"], label="Average Wealth")
plt.plot(model_data["Average_Savings"], label="Average Savings")  # Plot average savings
plt.xlabel("Time (months)")
plt.ylabel("Amount")
plt.legend()
plt.show()