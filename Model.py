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
            sigma: Risk aversion parameter (how much the agent dislikes uncertainty).
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

        # Update wealth
        self.wealth -= self.savings

         # --- Print V and g ---
        print(f"Agent {self.unique_id}: V =", V)
        print(f"Agent {self.unique_id}: g =", g)

         # --- Print wealth and savings before update ---
        print(f"Agent {self.unique_id}: Wealth before update =", self.wealth)
        print(f"Agent {self.unique_id}: Savings before update =", self.savings)

        # Find optimal savings for current wealth
        self.savings = g[np.argmin(np.abs(wealth_grid - self.wealth))]

        # Update wealth
        self.wealth -= self.savings

        # --- Print wealth and savings after update ---
        print(f"Agent {self.unique_id}: Wealth after update =", self.wealth)
        print(f"Agent {self.unique_id}: Savings after update =", self.savings)

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
        V_next = self.utility(consumption) + self.delta * V_old[np.argmin(np.abs(wealth_grid - possible_savings[:, np.newaxis]), axis=1)]  # Calculate the value function for all possible savings

        # Store current savings as previous_savings for the next step  # New line
        self.previous_savings = possible_savings[np.argmax(V_next)]  # New line

        return np.max(V_next), self.previous_savings  # Modified to return self.previous_savings



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

#_____________________________________________________________________ Example usage_______________________________________________________________________________
N = 1  # Number of agents
width = 10
height = 10
interest_rate = 0.1  # Annual interest rate
sigma = 0.4387 # Example risk aversion parameter

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
for i in range(12):  # Run for X months 
    model.step()

# Analyze data
model_data = model.datacollector.get_model_vars_dataframe()
agent_data = model.datacollector.get_agent_vars_dataframe()  # Get individual agent data

for agent_id, agent_df in agent_data.groupby("AgentID"):
    savings_list = np.ravel(agent_df["Savings"])
    print(f"Agent {agent_id} Savings:", savings_list)

plt.figure(figsize=(12, 6))

# Plot average wealth and savings for all agents over time
plt.subplot(2, 2, 1)
plt.plot(model_data["Average_Wealth"], label="Average Wealth")
plt.plot(model_data["Average_Savings"], label="Average Savings")  # Plot average savings
plt.xlabel("Time (months)")
plt.ylabel("Amount")
plt.legend()


# Plot individual agent wealth
plt.subplot(2, 2, 2) 
for agent_id, agent_df in agent_data.groupby("AgentID"):
    plt.plot(np.ravel(agent_df["Wealth"]), label=f"Agent {agent_id}")
plt.xlabel("Time (months)")
plt.ylabel("Wealth")
plt.title("Individual Agent Wealth Over Time")
plt.legend()


# Plot individual agent savings
plt.subplot(2, 2, 3) 
for agent_id, agent_df in agent_data.groupby("AgentID"):
    plt.plot(np.ravel(agent_df["Savings"]), label=f"Agent {agent_id}")
plt.xlabel("Time (months)")
plt.ylabel("Savings")
plt.title("Individual Agent Savings Over Time")
plt.legend()

plt.tight_layout()  # Adjust spacing between plots
plt.show()