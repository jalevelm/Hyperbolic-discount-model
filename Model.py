from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector  

import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize  # This line imports a tool for finding the best solution to a problem


class SavingAgent(Agent):
    """
    An agent with hyperbolic discounting preferences, making saving decisions.
    """
    def __init__(self, unique_id, model, beta, delta, init_wealth, salary, sigma):
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
        self.wealth = init_wealth  # Initialize with the first "salary"
        self.salary = salary
        self.sigma = sigma
        self.savings = 0 

    def step(self):
        """
        Define the agent's actions in each time step.
        """
        # Calculate threshold R* (using global interest_rate)
        interest_rate = self.model.interest_rate  
        R_star = 1 + (1 - self.delta) / (self.beta * self.delta)


# Optimization: This is where the agent decides how much to save
        # It uses a phyton tool called 'optimize.minimize_scalar' to find the best savings amount
        # This tool tries different savings amounts and calculates the "lifetime utility" for each one
        # "Lifetime utility" is like a score for how happy/satisfied the agent is with their spending and saving over their entire life
        # The tool finds the savings amount that gives the highest "lifetime utility" score
        result = optimize.minimize_scalar(
            lambda savings: -self.calculate_lifetime_utility(savings),  # Negative for maximization (because the tool is designed to find the minimum, we use a negative sign to make it find the maximum)
            bounds=(0, self.wealth),  # Savings cannot exceed wealth (the agent can't save more money than they have)
            method='bounded'  # This tells the tool to only look for solutions within the allowed bounds
        )
        self.savings = result.x  # Optimal savings (the best savings amount is stored in 'result.x', and we save it as the agent's savings)

        # Update wealth: This part updates the agent's wealth after saving and earning income
        self.wealth -= self.savings  # Subtract savings from wealth (the agent has less money now because they saved some)
        self.wealth += self.salary   # Add salary to wealth (the agent earns money from their job)

    
    def calculate_lifetime_utility(self, savings):
        """
        Calculate the agent's lifetime utility given a savings amount.
        """
        total_lifetime_utility = 0  # Initialize total utility
        current_wealth = self.wealth  # Start with the agent's current wealth

        # Calculate utility for a certain number of periods (e.g., 240 months)
        for t in range(120):  
            consumption = current_wealth - savings  # Calculate consumption for the current period
            
            # Calculate the discounted utility for the current period
            discounted_utility = self.beta * (self.delta ** t) * (consumption ** (1 - self.sigma) / (1 - self.sigma))  

            total_lifetime_utility += discounted_utility  # Add the discounted utility to the total

            # Update current_wealth for the next period
            current_wealth = (current_wealth - consumption) * (1 + self.model.interest_rate / 12)  # Assuming monthly interest

        return total_lifetime_utility




class SavingModel(Model):
    """
    A model with multiple saving agents.
    """
    def __init__(self, N, width, height, interest_rate, sigma, beta_ranges, delta_ranges, salary_dist):
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
            salary_dist: A list of tuples defining the salary distribution.
                         Each tuple is (probability, (min_salary, max_salary)).
        """
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)  # Torus grid
        self.schedule = RandomActivation(self)
        self.interest_rate = interest_rate
        self.sigma = sigma

        # Create agents
        for i in range(self.num_agents):
            # Sample beta and delta from ranges
            beta = random.choice(np.arange(beta_ranges[0], beta_ranges[1] + 0.1, 0.1)) 
            delta = random.choice(np.arange(delta_ranges[0], delta_ranges[1] + 0.1, 0.1)) 

            # Sample salary based on distribution
            rand_num = random.random()
            cumulative_prob = 0
            for prob, salary_range in salary_dist:  
                cumulative_prob += prob
                if rand_num <= cumulative_prob:
                    salary = random.randint(salary_range[0], salary_range[1])
                    init_wealth = salary  # Set initial wealth equal to salary
                    break

            # Create the agent with the chosen parameters
            a = SavingAgent(i, self, beta, delta, init_wealth, salary, sigma)
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
N = 100  # Number of agents
width = 10
height = 10
interest_rate = 0.1  # Annual interest rate
sigma = 0.8 # Example risk aversion parameter

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

# Create a model instance with the specified parameters
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