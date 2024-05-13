# Path-planning-using-reinforcement-learning


Path planning using search and sampling algorithms
Goal: Develop a mobile robot that must traverse a set of points in a map with obstacles, while avoiding collisions.
Note that the create_map.py script generates maps with traversable obstacles (“holograms”) since collision detection against each pixel makes Webots run too slowly.
Your path planning algorithm must use search-based algorithms, such as Dijkstra, A* and D*, and sampling-based algorithms, such as Rapidly-exploring random trees (RRT), with different heuristics, using the LiDAR readings. To simplify, you can use the e-puck robot from the practical classes, which is already equipped with this sensor.
Other ideas:
- Do your algorithms work if part/all of the map is unknown?


*Nosso Tema *
Path planning using reinforcement learning
Goal: Same as the previous topic, but the robot instead uses reinforcement learning (RL) models to perform the path planning, based on the LiDAR readings. You may explore more classic RL algorithms, such Q-learning and SARSA, or deep learning models, such as DQN, PPO and DDPG. For the DRL library, if you use Python, you can use Gymnasium, alongside the Stable-Baselines3 package (which contains many DRL models).
Other ideas:
- Do your algorithms work if part/all of the map is unknown?


Proximal Policy Optimization (PPO)
Soft Actor-Critic (SAC) 
Deep Deterministic Policy Gradient (DDPG)
State-Action-Reward-State-Action (SARSA)
Q-learning
Deep Q-Network (DQN) 


conda create -n GymEnv python=3.10 pytorch::pytorch torchvision torchaudio -c pytorch
