from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector  

import random
import numpy as np
import matplotlib.pyplot as plt

class SavingAgent(Agent):
    def __init__(self, unique_id, model, beta, delta, init_wealth, salary, sigma):
        super().__init__(unique_id, model)
        self.beta = beta
        self.delta = delta
        self.wealth = init_wealth  # Initialize with the first "salary"
        self.salary = salary
        self.sigma = sigma
        self.savings = 0 

    def step(self):
        # Calculate threshold R* (using global interest_rate)
        interest_rate = self.model.interest_rate  
        R_star = 1 + (1 - self.delta) / (self.beta * self.delta)


        # Simplified saving decision (no optimization yet)
        if interest_rate > R_star:  # interest_rate here
            # Save a portion of wealth
            self.wealth = self.wealth * (1 + (interest_rate - R_star) / 2)  # Arbitrary saving rate for now
            save_amount = self.wealth * (interest_rate - R_star) / 2
            self.wealth += save_amount
            self.savings = save_amount  # Store savings
        else:
             # Disave a portion of wealth, but prevent negative wealth
            disave_amount = min(self.wealth, self.wealth * (R_star - interest_rate) / 2)  # Limit dissaving to current wealth
            self.wealth -= disave_amount  # Subtract disave_amount from wealth
            self.savings = -disave_amount # Store dissavings as negative savings

        # Add salary
        self.wealth += self.salary


class SavingModel(Model):
    def __init__(self, N, width, height, interest_rate, sigma, beta_ranges, delta_ranges, salary_dist):
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)  # Torus grid
        self.schedule = RandomActivation(self)
        self.interest_rate = interest_rate
        self.sigma = sigma

        # Create agents
        for i in range(self.num_agents):
            # Sample beta and delta from ranges
            beta = random.choice(np.arange(beta_ranges[0], beta_ranges[1] + 0.1, 0.1))  # Inclusive of end
            delta = random.choice(np.arange(delta_ranges[0], delta_ranges[1] + 0.1, 0.1))  # Inclusive of end

            # Sample salary based on distribution
            rand_num = random.random()
            cumulative_prob = 0
            for prob, salary_range in salary_dist:  # Changed loop variable
                cumulative_prob += prob
                if rand_num <= cumulative_prob:
                    salary = random.randint(salary_range[0], salary_range[1])
                    init_wealth = salary  # Set initial wealth equal to salary
                    break

            a = SavingAgent(i, self, beta, delta, init_wealth, salary, sigma)
            self.schedule.add(a)

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
        self.datacollector.collect(self)
        self.schedule.step()

# Example usage
N = 100  # Number of agents
width = 10
height = 10
interest_rate = 0.1  # Annual interest rate
sigma = 0.8

# Example usage with ranges for beta and delta
beta_ranges = (0.1, 1)  # Beta range from 0 to 1
delta_ranges = (0.1, 0.9)  # Delta range from 0 to 0.9

# Example wealth and salary distribution (replace with your data)
wealth_salary_dist = [
    (0.037, (22404, 32000)),  
    (0.082, (14936, 22403)),  
    (0.318, (7468, 14935)),  
    (0.563, (0, 7468))   
]

model = SavingModel(N, width, height, interest_rate, sigma, beta_ranges, delta_ranges, wealth_salary_dist)

# Run the model
for i in range(120):  # Run for 120 months (10 years)
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