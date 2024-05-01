import gym 
from gym import spaces
import numpy as np
import pandas as pd
import os
import argparse

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from config import EXCEL_PATH

MODEL_PATH = "multi_energy_model.zip"

import logging

# Configure logging
logging.basicConfig(filename='C:\\Users\\cucas\\Documents\\UoM\\Year 3\\Project\\energy_management.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s', datefmt='%I:%M:%S %p')

class MultiEnergyEnv(gym.Env):
    total_runs = 0  # Class-level attribute to track the number of runs
    total_steps = 0  # Class-level attribute to track the total number of steps

    def __init__(self, EXCEL_PATH):
        super(MultiEnergyEnv, self).__init__()
        # Constants
        self.COP = 3.0  # Coefficient of performance for heat pump
        self.max_power = 5  # kW - max power for conversion processes
        self.battery_capacity = 200  # kWh
        self.charge_efficiency = 0.9
        self.discharge_efficiency = 0.9
        self.degradation_per_cycle = 0.9
        self.alpha = 0.01  # Penalty unmet demand
        self.beta = 0.001   # Weight for energy cost
        self.current_timestep = 0  # Track timesteps within a day
        self.day_price = 0.10  # Higher price during the day
        self.night_price = 0.03  # Lower price at night
        self.price_schedule = [self.night_price] * 16 + [self.day_price] * 32 # 48 timesteps in a day (30-minute intervals for 24 hours)
        self.saved_costs = 0 # Initialize saved costs
        self.income_from_energy_sold = 0 #Initialize total Income
        self.delta = 1.5 # Weight for the income 
        self.total_energy_cost_accum = 0
        self.income_from_energy_sold_accum = 0
        self.unmet_electricity_accum = 0
        self.unmet_heat_accum = 0
        self.reward_accum = 0
        self.log_interval = 1000
        self.interval_step_count = 0

        # Load demand profiles
        self.demand_data = pd.read_excel(EXCEL_PATH, sheet_name='Building_27', engine='openpyxl')
        logging.info(self.demand_data.head())  # Print the first 5 rows of the DataFrame to inspect
        logging.info(self.demand_data.columns) # Print column names
        logging.info(self.demand_data.dtypes) # Check data types

        # Observation space: [Electricity Demand, Heat Demand, Battery Level]
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([np.inf, np.inf, self.battery_capacity]), dtype=np.float32)
        
        # Action space: [Use grid, Use battery, Charge battery, Generate heat]
        self.action_space = spaces.Discrete(5)

        # Initial state
        self.state = None
        self.reset()

    def step(self, action):
        MultiEnergyEnv.total_steps += 1
        electricity_demand, heat_demand, battery_level = self.state
        electricity_used = 0
        heat_generated = 0
        total_energy_cost = 0
        income_from_energy_sold = 0
        unmet_electricity = electricity_demand
        unmet_heat = heat_demand
        #logging.info(f"Action taken at step {self.current_step}: {action}")

        electricity_demand = self.demand_data['Electricity (kWh)'].iloc[self.current_step]
        heat_demand = self.demand_data['Heat (kWh)'].iloc[self.current_step]
        #logging.info(f"Step: {self.current_step}, Electricity Demand: {electricity_demand}, Heat Demand: {heat_demand}")

        # Determine current price based on time of day
        current_time_index = self.current_timestep % 48
        current_price = self.night_price if current_time_index < 16 else self.day_price

        self.current_step += 1
        done = self.current_step >= len(self.demand_data) - 1  # Ensure it doesn't go out of index

        # Action logic
        if action == 0:  # Use grid
            cost = electricity_used * current_price
            total_energy_cost += cost  # Add to total cost
            #print(f"Grid used: electricity_used={electricity_used}, cost={cost}")
       
        elif action == 1:  # Use battery           
            electricity_used = min(electricity_demand, battery_level * self.discharge_efficiency) # Calculate electricity used from battery
            battery_level -= electricity_used / self.discharge_efficiency # Reduce the battery electricity used, accounting for discharge efficiency
            cost = 0 # Since the cost of using the battery is considered zero because it was paid during charging
            current_time_index = self.current_timestep % 48  # Check current time
            if 0 <= current_time_index < 16: # Determine current electricity price
                current_price = self.night_price
            else:  # Day price applies for the rest of the day
                current_price = self.day_price           
            saved_cost = electricity_used * current_price # Calculate how much would have been spent if the same amount of electricity was bought from the grid
            self.saved_costs += saved_cost  # Save the saved cost to check at the end of the simulation.
            # print(f"Saved cost by using battery this step: ${saved_cost:.2f}")

        elif action == 2:  # Charge battery 
            current_time_index = self.current_timestep % 48  # Check current time in the daily cycle 
            if 0 <= current_time_index < 16: # Determine current electricity price based on time
                current_price = self.night_price
            else:  
                current_price = self.day_price           
            if battery_level < self.battery_capacity: # Check capacity to charge the battery
                charge_amount = min(self.battery_capacity - battery_level, self.max_power) # Calculate maximum possible charge
                battery_level += charge_amount * self.charge_efficiency # Increase the battery level possible               
                cost = charge_amount * current_price # Calculate the cost of charging
                total_energy_cost += cost # Add this cost to the total energy cost for the step
       
        elif action == 3: # Sell energy to the grid
            sell_amount = min(battery_level / self.discharge_efficiency, self.max_power)  # Determine amount to sell
            current_price = self.price_schedule[self.current_timestep % 48]
            income_from_energy_sold = sell_amount * current_price  # Calculate income
            battery_level -= sell_amount * self.discharge_efficiency  # Update battery level
             # Optionally, you could track cumulative income from selling energy
             #self.income_from_energy_sold += income_from_energy_sold
        
        elif action == 4:  # Generate heat
            heat_generated = min(heat_demand, self.max_power * self.COP)
            # Determine the current price based on the time of day
            current_time_index = self.current_timestep % 48
            if 0 <= current_time_index < 16:  # Night time pricing
                current_price = self.night_price
            else:  # Day time pricing
                current_price = self.day_price   
            # Calculate the cost of generating heat
            cost = heat_generated * current_price  # Adjusting to use the amount of heat generated
            total_energy_cost += cost

        # Calculate unmet demands
        unmet_electricity = max(0, electricity_demand - electricity_used)
        unmet_heat = max(0, heat_demand - heat_generated)

        # Reward function
        reward = ((-self.beta * total_energy_cost + self.delta * income_from_energy_sold - self.alpha * (unmet_electricity + unmet_heat))/2)

    # Update accumulators
        self.total_energy_cost_accum += total_energy_cost
        self.income_from_energy_sold_accum += income_from_energy_sold
        self.unmet_electricity_accum += max(0, electricity_demand - electricity_used)
        self.unmet_heat_accum += max(0, heat_demand - heat_generated)
        self.reward_accum += reward
        
        self.interval_step_count +=1

        if self.interval_step_count >= self.log_interval:
            # Log the averages
            logging.info(f"Step: {MultiEnergyEnv.total_steps}")
            logging.info(f"Total Energy Cost: {self.total_energy_cost_accum}")
            logging.info(f"Total Income from Energy Sold: {self.income_from_energy_sold_accum}")
            logging.info(f"Total Unmet Electricity Demand: {self.unmet_electricity_accum}")
            logging.info(f"Total Unmet Heat Demand: {self.unmet_heat_accum}")
            logging.info(f"Total Reward: {self.reward_accum}")

            # Reset accumulators
            self.total_energy_cost_accum = 0
            self.income_from_energy_sold_accum = 0
            self.unmet_electricity_accum = 0
            self.unmet_heat_accum = 0
            self.reward_accum = 0
            self.interval_step_count = 0

        self.current_timestep += 1

        # Update the state
        if done:
            logging.info(f"End of data reached at step {self.current_step}. Episode will reset.")
            return np.array(self.state), reward, done, {}
        #print(f"Step: {MultiEnergyEnv.total_steps}, Total Runs: {MultiEnergyEnv.total_runs}")  # Print the current step and run count

        self.current_timestep += 1
        self.state = (electricity_demand, heat_demand, battery_level)

        return np.array(self.state), reward, done, {}

    def reset(self):

        # Reset episode-specific variables
        self.current_step = 0
        self.current_timestep = 0  # Reset time to midnight if it affects your simulation

        # Reset accumulators if used
        self.total_energy_cost_accum = 0
        self.income_from_energy_sold_accum = 0
        self.unmet_electricity_accum = 0
        self.unmet_heat_accum = 0
        self.reward_accum = 0
        
        # Retrieve initial conditions for the new episode
        initial_electricity = self.demand_data['Electricity (kWh)'].iloc[0]
        initial_heat = self.demand_data['Heat (kWh)'].iloc[0]
        self.state = (initial_electricity, initial_heat, self.battery_capacity)

         # Log the reset to help with debugging and tracking the environment's behavior
        logging.info(f"Environment reset. Starting new episode. Total runs: {MultiEnergyEnv.total_runs}")
        #print(f"Environment reset. Starting new episode. Total runs: {MultiEnergyEnv.total_runs}")
        MultiEnergyEnv.total_runs += 1

        return np.array(self.state)
    
    def render(self, mode='human'):
        pass

def get_args():
    parser = argparse.ArgumentParser(description='Train a new model or continue training an existing one.')
    parser.add_argument('--continue', dest='continue_training', action='store_true', # Write python CodeRL.py --continue
                        help='Continue training from an existing model')
    parser.add_argument('--restart', dest='continue_training', action='store_false', # Write python CodeRL.py --restart
                        help='Restart training with a new model')
    parser.set_defaults(continue_training=False)
    return parser.parse_args()

def load_or_initialize_model(env, model_path, continue_training):
    if continue_training and os.path.exists(model_path):
        print("Loading existing model.")
        return PPO.load(model_path, env)
    else:
        print("Initializing new model.")
        return PPO("MlpPolicy", env, verbose=1)

def evaluate_model(env, model):
    obs = env.reset()
    total_reward = 0
    num_steps = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        num_steps += 1
        if done:
            break
    print("Evaluation completed.")
    print("Total reward:", total_reward)
    
def main():
    args = get_args()
    env = DummyVecEnv([lambda: MultiEnergyEnv(EXCEL_PATH)])  # Handles automatic reset on done
    model = load_or_initialize_model(env, MODEL_PATH, args.continue_training)

    total_episodes = 1  # More episodes for more extensive training
    for episode in range(total_episodes):
        obs = env.reset()
        done = False
        total_rewards = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _info = env.step(action)
            total_rewards += reward

        if (episode + 1) % 100 == 0:  # Save the model every 100 episodes
            model.save(MODEL_PATH)
            print(f"Episode {episode + 1}: Total rewards = {total_rewards}")
            print(f"Model saved at episode {episode + 1}")

    # Save the model after the final episode
    model.save(MODEL_PATH)
    print("Training completed. Model saved to", MODEL_PATH)
    print("Excel Sheet name: Building_27")

    # Evaluate the model after training is complete
    evaluate_model(env, model)

if __name__ == "__main__":
    main()

def train_agent():
    env = DummyVecEnv([lambda: MultiEnergyEnv(EXCEL_PATH)])  
    model = DQN('MlpPolicy', env, verbose=1, learning_rate=0.1, gamma=0.95, exploration_final_eps=0.1, exploration_initial_eps=0.99, batch_size=64)
    model.learn(total_timesteps=17520)  
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

def evaluate_agent():
    env = DummyVecEnv([lambda: MultiEnergyEnv(EXCEL_PATH)])  
    model = DQN.load(MODEL_PATH, env=env)
    obs = env.reset()
    total_reward = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done.any():
            break
    print("Total reward:", total_reward)
    
def inspect_model_parameters(model):
    print("Model's policy network:", model.policy)
    print("Model's parameters:")
    for name, param in model.policy.named_parameters():
        print(name, param.data)