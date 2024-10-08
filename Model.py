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

    def step(self):
        # Calculate threshold R* (using global interest_rate)
        annual_interest_rate = self.model.interest_rate
        monthly_interest_rate = (1 + annual_interest_rate)**(1/12) - 1
        R_star = 1 + (1 - self.delta) / (self.beta * self.delta)


        # Simplified saving decision (no optimization yet)
        if self.model.interest_rate > R_star:
            # Save a portion of wealth
            self.wealth = self.wealth * (1 + (self.model.interest_rate - R_star) / 2)  # Arbitrary saving rate for now
        else:
            # Disave a portion of wealth
            self.wealth = self.wealth * (1 - (R_star - self.model.interest_rate) / 2)  # Arbitrary dissaving rate for now

        # Add salary
        self.wealth += self.salary

class SavingModel(Model):
    def __init__(self, N, width, height, interest_rate, sigma, beta_dist, delta_dist, init_wealth_dist, salary_dist):
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)  # Torus grid
        self.schedule = RandomActivation(self)
        self.interest_rate = interest_rate / 12  # Convert annual to monthly
        self.sigma = sigma

        # Create agents
        for i in range(self.num_agents):
            # Sample from distributions
            beta = random.choice(beta_dist)
            delta = random.choice(delta_dist)
            init_wealth = random.choice(init_wealth_dist)
            salary = random.choice(salary_dist)

            a = SavingAgent(i, self, beta, delta, init_wealth, salary, sigma)
            self.schedule.add(a)

            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        # Data  collection
        self.datacollector = DataCollector(
            model_reporters={"Average_Wealth": lambda m: np.mean([a.wealth for a in m.schedule.agents])},
            agent_reporters={"Wealth": "wealth"}
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

# Example usage
N = 100  # Number of agents
width = 10
height = 10
interest_rate = 0.05  # Annual interest rate
sigma = 0.8

# Example distributions (replace with your country-specific data)
beta_dist = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # Example distribution
delta_dist = [0.9, 0.95, 0.98]  # Example distribution
init_wealth_dist = list(range(1000, 2000))  # Example distribution
salary_dist = list(range(1000, 1500))  # Example distribution

model = SavingModel(N, width, height, interest_rate, sigma, beta_dist, delta_dist, init_wealth_dist, salary_dist)

# Run the model
for i in range(120):  # Run for 120 months (10 years)
    model.step()

# Analyze data
agent_data = model.datacollector.get_agent_vars_dataframe()
model_data = model.datacollector.get_model_vars_dataframe()

# Plot average wealth over time
plt.plot(model_data["Average_Wealth"])
plt.xlabel("Time (months)")
plt.ylabel("Average Wealth")
plt.show()