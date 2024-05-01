# 1. Reainforcemenet-Learning-in-Multi-Energy-Systems
2. Introduction
This repository contains two Reinforcement Learning (RL) models: a Q-learning model and more advanced models using Deep Q-
Networks (DQN) and Proximal Policy Optimization (PPO). These models are designed to optimize energy management in a
simulated building environment, focusing on efficient electricity and heat usage through dynamic decision-making. The
project aims to demonstrate the practical applications of RL in managing complex multi-energy systems, improving operational
efficiency, and reducing costs.

3. Install required packages:
  in the bash write the following:
  "pip install -r requirements.txt"
  All necessary libraries, are gym, numpy, pandas, matplotlib, and stable-baselines3 if using DQN or PPO.

4. To run either the Q-learning or DQN/PPO models, you'll need to ensure your setup correctly includes an Excel file for
   input data, which the models use to simulate different energy demands. Here's a step-by-step guide:

  Prepare Your Data:
  Ensure you have an Excel file with the required format. The default expected file is EXCEL_PATH as defined in config.py.
  Update the config.py file if your Excel file name or path differs from the default. This file contains constants and  
  configurations used throughout the project.

5. More Technical Details
  Reward Function
  Both the Q-learning and the DQN/PPO models use a reward function that calculates the efficiency of energy use and cost-
  effectiveness at each step in the environment. The function is formulated as follows:

  𝑅𝑡𝑜𝑡𝑎𝑙=(−𝛽×Total Energy Cost+𝛿×Total Income from Energy Sold−𝛼×(Unmet Electricity Demand+Unmet Heat Demand))/2
  
  Where:
  𝛽: penalizes higher energy costs.
  𝛿: rewards income generated from selling excess energy.
  𝛼: penalizes unmet energy demands.
  The division by 2 normalizes the reward for computational feasibility.
  This reward structure helps guide the RL model towards not only minimizing costs but also maximizing the usage efficiency
  of the available resources and meeting the building's energy demands as closely as possible.
  The detailed setup for each model ensures the agent interacts with a complex environment in a way that optimizes energy
  management in buildings, thus offering insights into scalable and adaptable solutions for real-world energy systems.

6. Known Issues/Future Improvements:
  Q-learning: May converge to suboptimal policies in very complex environments due to its simplistic nature. Future
  improvements could involve integrating function approximators to handle larger state spaces.
  DQN/PPO: Require substantial computational resources and can be sensitive to hyperparameter settings. Future iterations
  could explore more efficient neural network architectures or advanced exploration strategies to enhance learning speed
  and stability.
  Future work will also include expanding the environmental model to incorporate additional renewable energy sources and
  exploring real-time pricing models to increase realism and applicability to real-world scenarios.

7. Check the code version it is written nicer and clearer. Go to the top left
