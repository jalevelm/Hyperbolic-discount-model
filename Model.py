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
        print(f"Agent {unique_id} initialized. init_wealth: {self.init_wealth}, wealth: {self.wealth}")  # Debug print

    def step(self):
        """
        Define the agent's actions in each time step.
        """
        if self.step_count == 0:
            print(f"Agent {self.unique_id}: First Step - Beta: {self.beta}, Delta: {self.delta}, R_star: {self.R_star}") # Debug print
        self.step_count += 1

        print(f"Agent {self.unique_id}: Step start. Wealth: {self.wealth}, Previous Savings: {self.previous_savings}") # Debug print

        wealth_grid = self.model.wealth_grid
        k = self.wealth

        # Initialize value function and saving function
        V = np.zeros_like(wealth_grid)
        g = np.zeros_like(wealth_grid)
        print(f"Agent {self.unique_id}: Starting value iteration...")

        # Value iteration and policy improvement
        # Initialize V_old to something different from V to ensure at least one iteration
        V_old = np.ones_like(wealth_grid) * np.inf
        iterations = 75
        
        for _ in range(iterations): # Adjust number of iterations as needed
            try:
                # Parallelize the loop over wealth_grid
                results = Parallel(n_jobs=-1)(
                    delayed(self.optimize_savings)(k_val, V_old, wealth_grid)
                    for k_val in wealth_grid
                )
                #Check for None values from the try-except block
                V = np.array([result[0] if result is not None else 0 for result in results])
                g = np.array([result[1] if result is not None else 0 for result in results])

            except Exception as e:
                print(f"Agent {self.unique_id}: Error in parallel loop (iteration {_}): {e}")
                raise

            # Convergence check
            if np.max(np.abs(V - V_old)) < 1e-3:  # adjust tolerance
                print(f"Agent {self.unique_id}: Value function converged after {_} iterations.")
                break
            V_old = V.copy() # Copy V to V_old *after* the iteration
        else:
            print(f"Agent {self.unique_id}: Value function DID NOT converge after {iterations} iterations.")

        closest_index = np.argmin(np.abs(wealth_grid - self.wealth))
        self.savings = g[closest_index]
        self.wealth = self.wealth + (self.model.interest_rate - 1) * self.savings
        self.previous_savings = self.savings

        print(f"Agent {self.unique_id}: Step end.  Wealth: {self.wealth}, Savings: {self.savings}") # Debug print


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
        if self.sigma == 1:
            return np.log(consumption)
        return consumption**(1 - self.sigma) / (1 - self.sigma) if consumption > 0 else -np.inf

    
    def optimize_savings(self, k, V, wealth_grid):
        """Optimize savings for a given wealth level k."""
        try:
            beta = self.beta
            consumption = wealth_grid - k
            consumption[consumption <= 0] = 1e-10  # Avoid log(0)

            future_wealth = self.wealth + (self.model.interest_rate - 1) * k

            # --- KEY CHANGE: Input validation and clipping ---
            future_wealth_valid = np.isfinite(future_wealth)  # Check for inf/nan
            if not np.all(future_wealth_valid):
                #print(f"Agent {self.unique_id}: Invalid future_wealth: {future_wealth}") #Debugging
                future_wealth = np.where(future_wealth_valid, future_wealth, np.nanmax(future_wealth[future_wealth_valid])) # Replace inf with max finite value

            # Clip future_wealth to a reasonable range
            min_wealth = wealth_grid[0] - (wealth_grid[1] - wealth_grid[0])  # One grid step below min
            max_wealth = wealth_grid[-1] + (wealth_grid[1] - wealth_grid[0])  # One grid step above max
            future_wealth = np.clip(future_wealth, min_wealth, max_wealth)
            # --- END KEY CHANGE ---


            future_utility = beta * np.interp(future_wealth, wealth_grid, V, left=V[0], right=V[-1]) #Extrapolate by using the extreme values
            utility = np.log(consumption) + future_utility
            return -utility.sum(), k  # Negative for minimization, return k as well

            print(f"    k_next: {k_next:.2f}, consumption: {consumption:.2f}, future_utility: {future_utility:.2f}, total_utility: {total_utility:.2f}") 

        except Exception as e:
            print(f"Error in optimize_savings with k={k}: {e}")
            return None, None  # Return None if optimization fails


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
        self.wealth_grid = np.linspace(0, self.max_wealth * self.interest_rate, 200) #ajust wealth grid size


        # Data collection 
        self.datacollector = DataCollector(
            agent_reporters={
                "Wealth": "wealth",
                "Savings": "savings",
                "Consumption": lambda a: a.wealth - a.savings,
                "R_star": "R_star",
                "Interest_Rate": lambda a: a.model.interest_rate,
                "Utility": lambda a: a.utility(a.wealth - a.savings),
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

interest_rate = 1.16
sigma = 0.4387 # elasticity of satisfaction
wealth_dist = [
    (0.41520521, (0, 9999)),  # Less than 10,000 USD
    (0.477205824, (10000, 99999)),  # Between 10,000 and 100,000 USD
    (0.103122194, (100000, 999999)),  # Between 100,000 and 1,000,000 USD
    (0.004466772, (1000000, 10000000))  # More than 1,000,000 USD 
]


# -------------------------------------------------------- Run Simulation-----------------------
profile_to_use = "impulsive"
model = SavingModel(agent_profiles[profile_to_use], interest_rate, sigma, wealth_dist)

for i in range(2):
    print(f"--- Starting model step {i+1} ---")  # Debug print
    try:
        model.step()
    except Exception as e:
        print(f"Error in model step {i+1}: {e}")
        break  # Stop the loop if there's an error
    print(f"--- Finished model step {i+1} ---")  # Debug print

# Analyze data
try: 
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

except Exception as e:
    print(f"Error during data analysis or plotting: {e}")