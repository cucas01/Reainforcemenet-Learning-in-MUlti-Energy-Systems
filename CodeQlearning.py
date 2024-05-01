import numpy as np
import pandas as pd
import gym
from gym import spaces
import os
import logging
from config import EXCEL_PATH

import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(filename='energy_managementQ.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s', datefmt='%I:%M:%S %p')

class MultiEnergyEnv(gym.Env):
    
    def __init__(self, EXCEL_PATH):
        super().__init__()
        self.demand_data = pd.read_excel(EXCEL_PATH, sheet_name='Building_27', engine='openpyxl')
        self.n_steps = len(self.demand_data)
        self.current_step = 0

        # Initialize saved costs
        self.saved_costs = 0

        # Define action and state space
        self.action_space = spaces.Discrete(5)  # Actions: grid, battery, charge, sell, heat
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([np.inf, np.inf, 200]), dtype=np.float32)

        test_actions = [self.action_space.sample() for _ in range(1000)]
        assert all(0 <= a < 5 for a in test_actions), "Action space is compromised."

        # Constants
        self.COP = 3.0  # Coefficient of performance for the heat pump
        self.max_power = 5  # max power kW
        self.battery_capacity = 200  # kWh
        self.charge_efficiency = 0.9
        self.discharge_efficiency = 0.9
        self.alpha = 0.3  # Penalty for unmet demand
        self.beta = 0.08   # Weight for energy cost
        self.delta = 1.1  # Weight for the income
        self.night_price = 0.03
        self.day_price = 0.10
        self.price_schedule = [self.night_price]*16 + [self.day_price]*32

        # Q-table
        self.q_table = np.zeros((10, 10, 10, 5))  # Ensure dimensions match state space
        # Hyperparameters
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration probability
        self.learning_rate_decay = 0.99  # Optional: Decay for learning rate

        self.reset()

    def reset(self):
        initial_electricity = self.demand_data['Electricity (kWh)'].iloc[0]
        initial_heat = self.demand_data['Heat (kWh)'].iloc[0]
        initial_battery_level = self.battery_capacity
        self.state = self.discretize_state(initial_electricity, initial_heat, initial_battery_level)
        return np.array(self.state)

    def step(self, action):
        current_state = self.state
        new_state, reward, done = self.simulate_action(current_state, action)
        self.state = self.discretize_state(*new_state)  # Ensure new_state is a tuple (electricity, heat, battery)
        return np.array(self.state), reward, done, {}

    def simulate_action(self, current_state, action):
        electricity_demand, heat_demand, battery_level = current_state
        current_price = self.price_schedule[self.current_step % 48]  # Get price based on the current timestep

    # Initialize variables for changes
        electricity_used = 0
        heat_generated = 0
        total_energy_cost = 0
        income_from_energy_sold = 0

    # Action logic
        if action == 0:  # Use grid
            electricity_used = electricity_demand  # Simplified assumption
            cost = electricity_used * current_price
            total_energy_cost += cost

        elif action == 1:  # Use battery
            if battery_level > 0:
                electricity_used = min(electricity_demand, battery_level * self.discharge_efficiency)
                battery_level -= electricity_used / self.discharge_efficiency
                saved_cost = electricity_used * current_price
                self.saved_costs += saved_cost

        elif action == 2:  # Charge battery
            charge_amount = min(self.battery_capacity - battery_level, self.max_power)
            battery_level += charge_amount * self.charge_efficiency
            cost = charge_amount * current_price
            total_energy_cost += cost

        elif action == 3:  # Sell energy to the grid
            sell_amount = min(battery_level / self.discharge_efficiency, self.max_power)
            income_from_energy_sold = sell_amount * current_price * 1.1
            battery_level -= sell_amount * self.discharge_efficiency

        elif action == 4:  # Generate heat
            heat_generated = min(heat_demand, self.max_power * self.COP)
            cost = heat_generated * current_price
            total_energy_cost += cost

        # Calculate unmet demands
        unmet_electricity = max(0, electricity_demand - electricity_used)
        unmet_heat = max(0, heat_demand - heat_generated)

        # Reward function (simplified for this example)
        reward = -self.alpha * (unmet_electricity + unmet_heat) - self.beta * total_energy_cost + self.delta * income_from_energy_sold

        # Determine if the episode is done
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1

        # Update state
        new_state = (electricity_demand, heat_demand, battery_level)

        return new_state, reward, done
    
    def discretize_state(self, electricity, heat, battery):
        # Assuming you define ranges or bins for each state variable
        max_electricity = self.demand_data['Electricity (kWh)'].max()
        max_heat = self.demand_data['Heat (kWh)'].max()
        electricity_bins = np.linspace(0, max_electricity, num=10)  # Adjust num for finer granularity
        heat_bins = np.linspace(0, max_heat, num=10)
        battery_bins = np.linspace(0, self.battery_capacity, num=10)

        # Digitize (find bin indices)
        electricity_idx = min(np.digitize(electricity, electricity_bins) - 1, 9)  # 9 is the last valid index for 10 bins
        heat_idx = min(np.digitize(heat, heat_bins) - 1, 9)
        battery_idx = min(np.digitize(battery, battery_bins) - 1, 9)


        return (electricity_idx, heat_idx, battery_idx)

    def render(self, mode='human'):
        # Optional: Implement visualization
        pass

    def choose_action(self, state):
        return self.action_space.sample()
    
#def plot_q_table(q_table):
    # Heatmap of the Q-table
    #ax = sns.heatmap(q_table, annot=True, cmap='coolwarm')
    #ax.set(title='Heatmap of Q-Table', xlabel='Actions', ylabel='States')
    #plt.show()

def main():
    env = MultiEnergyEnv(EXCEL_PATH)
    episodes = 17520
    total_rewards = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = env.choose_action(state)

            if not (0 <= action < env.action_space.n):
                print(f"Detected out-of-bound action before assertion: {action}")
                continue  # Skip this iteration or handle it as needed

            assert 0 <= action < env.action_space.n, "Action index out of bounds"

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            if np.any(np.isnan(env.q_table)) or np.any(np.isinf(env.q_table)):
                print("NaN or Inf values detected in Q-table.")

            # Update Q-table using the learning rate
            old_value = env.q_table[state[0], state[1], state[2], action]
            next_max = np.max(env.q_table[next_state[0], next_state[1], next_state[2]])
            new_value = (1 - env.learning_rate) * old_value + env.learning_rate * (reward + env.discount_factor * next_max)
            env.q_table[state[0], state[1], state[2], action] = new_value
            state = next_state  # Update the state for the next iteration

            # Update epsilon and learning rate
            env.epsilon = max(env.epsilon * env.epsilon_decay, env.epsilon_min)
            env.learning_rate *= env.learning_rate_decay # Optional
        total_rewards.append(episode_reward)      

        if episode % 100 == 0:
            print(f'Episode: {episode}, Reward: {episode_reward}, State: {state}')
    grand_total_reward = sum(total_rewards)
    print(f"Total reward {grand_total_reward}")
    print(f"Excel sheet name: Buidling_27")
    plt.plot(total_rewards)
    plt.title('Zoomed in Reward Obtained Per Step')
    plt.xlabel('Steps')
    plt.ylabel('Total Reward')
    plt.show()


if __name__ == "__main__":
    main()