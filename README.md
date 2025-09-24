```
# Saving and Dissaving with Hyperbolic Discounting - Agent-Based Model

This repository contains an agent-based model (ABM) simulating saving and dissaving behavior based on the principles of hyperbolic discounting. The model is inspired by and implements concepts from the paper "Saving and Dissaving with Hyperbolic Discounting" by Cao and Werning (2018).

## Model Overview

This ABM simulates a population of agents who make saving and dissaving decisions based on:

- **Hyperbolic Discounting:** Agents exhibit present bias, meaning they value immediate consumption more than future consumption. This is modeled using the `beta` (present bias) and `delta` (standard discount factor) parameters.
- **Interest Rate:**  The prevailing interest rate in the economy influences agent decisions.
- **Social Interactions:** Agents can share financial information and compare their saving behavior with their neighbors, impacting their choices.
- **Network Structure:** Agents are connected in a small-world network, representing realistic social structures.
- **Heterogeneous Agents:** Agents have diverse characteristics, including varying levels of present bias, income, and financial literacy.

## Key Features

- **Optimization:** The model implements the optimization problem described in the Cao and Werning paper. Agents choose saving/dissaving amounts to maximize their lifetime utility, considering their present bias and the interest rate.
- **Utility Function:** The model uses an isoelastic utility function, consistent with the paper, to represent agent preferences.
- **Numerical Optimization:**  The model uses numerical optimization techniques to find the saving amount that maximizes the agent's lifetime utility.
- **Markov Equilibria Approximation:** The model approximates the Markov equilibria described in the paper by iteratively calculating lifetime utility and finding the saving amount that maximizes it.

## Files

- `model.py`: Contains the Python code for the agent-based model, including the `SavingAgent` and `SavingModel` classes.
- `README.md`: This file, providing a description of the model and instructions.

## How to Run

1. **Install Requirements:** Make sure you have the necessary Python packages installed. 
2. **Run the Model:** Execute the `model.py` file. 
## Key Parameters

- `N`: Number of agents in the model.
- `avg_node_degree`: Average number of connections per agent in the network.
- `interest_rate`: Global interest rate in the economy.
- `sigma`: Parameter controlling the curvature of the utility function.
- `beta_ranges`: Range of possible values for the present bias parameter (beta).
- `delta_ranges`: Range of possible values for the standard discount factor (delta).
- `wealth_salary_dist`: Distribution of salaries for the agents.

