# Portfolio
## Reinforcement learning projects

### Playing Space Invaders with an actor-critic PPO algorithm
I have trained an actor-critic agent with a [Proximal policy optimization algorithm](https://arxiv.org/pdf/1707.06347.pdf) to play the Atari 2600 game Space Invaders using the reinforcement learning library [TF-Agents](https://www.tensorflow.org/agents). Using the [OpenAI Gym environment](https://gym.openai.com/envs/SpaceInvaders-ram-v0/),  the agent has been trained using as input the RAM of the Atari machine consisting of (only!) 128 bytes. In this environment what the agent "sees" is not the rendered image showing the space ships, projectiles and shields but just a sequence of 128 integer numbers corresponding to the RAM containing the stored information that represents the game state. The agent learns to consistently dodge projectiles and is able to complete the first level of the game.

<p align="center">
<img src="./Notebooks/Space_Invaders_episode.gif" alt="An episode played by the agent" width="240" height="336"> <br>
An episode played by the trained agent
</p>

[![Jupyter](https://img.shields.io/badge/Jupiter-View%20Notebook-orange?&logo=Jupyter)](https://nbviewer.jupyter.org/url/GabrieleSgroi.github.io/Notebooks/PPO.ipynb)

### Playing MS-Pacman with a categorical DQN
I have trained a [Categorical Deep Q-Network ](https://arxiv.org/pdf/1707.06887.pdf) to play the Atari 2600 game MsPacman using the reinforcement learning library [TF-Agents](https://www.tensorflow.org/agents). <br> Using the [OpenAI Gym environment](https://gym.openai.com/envs/MsPacman-ram-v0/),  the agent has been trained using as input the RAM of the Atari machine consisting of (only!) 128 bytes. In this environment what the agent "sees" is not the rendered image showing the maze, dots, and ghosts but just a sequence of 128 integer numbers corresponding to the RAM containing the stored information that represents the game state. The agent learns to consistently navigate the maze and to chase the ghosts after having eaten the power pellets.

<p align="center">
<img src="./Notebooks/pacman_episode.gif" alt="An episode played by the agent" width="240" height="336"> <br>
An episode played by the trained agent
</p>


[![Jupyter](https://img.shields.io/badge/Jupiter-View%20Notebook-orange?&logo=Jupyter)](https://nbviewer.jupyter.org/github/GabrieleSgroi/GabrieleSgroi.github.io/blob/main/Notebooks/Pacman_Categorical_DQN.ipynb)

## Deep learning projects: Unsupervised learning

### Hierarchical Vector Quantized Variational Autoencoder for image generation (VQ-VAE)
I have implemented a custom architecture of a hierarchical vector quantized variational autoencoder (VQ-VAE) following the concept introduced in the paper [Generating Diverse High-Fidelity Images with VQ-VAE-2](https://arxiv.org/pdf/1906.00446.pdf) togheter with custom implementations of the PixelCNN priors introduced in the paper [Conditional Image Generation with PixelCNN Decoders](https://arxiv.org/pdf/1606.05328.pdf). The architectures of the models were customized in order to retain good performance on large resolution (512x512) images while remaining light enough to train on free Kaggle/Colab TPUs and GPUs. The model has been trained on the image data of the Kaggle competition [Humpback Whale Identification](https://www.kaggle.com/c/humpback-whale-identification) as this dataset offered a reasonable number of high resolution images. 

<p align="center">
<img src="https://github.com/GabrieleSgroi/hierarchical-VQ-VAE/blob/main/Selected%20images/2f07f6d7-b754-435c-8d48-969ec3ed3985.jfif" alt="Image" width="256" height="256"> <br>
An image generated by the model
</p>

[![GitHub](https://img.shields.io/badge/Github-View%20on%20GitHub-blue?&logo=github)](https://github.com/GabrieleSgroi/hierarchical-VQ-VAE)
[![Colab](https://img.shields.io/badge/Colab-View%20example%20notebook-blue?&logo=googlecolab)](https://colab.research.google.com/drive/1zLrX5q5zKA6dCbOWpepagYSYLzDNbc9v?usp=sharing)

## Deep learning projects: Supervised learning 
### Vehicle motion prediction with 3d CNN 
I have trained a 3d convolutional neural network to predict the future trajectories of vehicles as an entry for the [Lyft Motion Prediction for Autonomous Vehicles](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles) competition on Kaggle. The model takes as input few frames of bird's-eye view images containing the visual representation of all the vehicles in the scene and, through intermediate layers including also 3d convolutions (1 temporal+ 2 spatial dimensions), predicts 3 possible future trajectories for the target vehicle and associate a probability to each of them. The model easily outperforms the baseline benchmark set by the competition after being trained on approximately 10% of the training set. 

[![Jupyter](https://img.shields.io/badge/Jupiter-View%20Notebook-orange?&logo=Jupyter)](https://nbviewer.jupyter.org/url/GabrieleSgroi.github.io/Notebooks/lyft-vehicles-motion-prediction-3d-cnn-with-keras.ipynb)

### Image segmentation to identify glomeruli in Kidney

I trained a [U-Net](https://arxiv.org/pdf/1505.04597.pdf) like architecture to predict segmentation masks in order to identify glomeruli inspired by the task of the [HuBMAP - Hacking the Kidney](https://www.kaggle.com/c/hubmap-kidney-segmentation) competition on Kaggle. The model adds an attention mechanism to the U-Net through the use of the [Convolutional Block Attention Module](https://arxiv.org/pdf/1807.06521.pdf). The model was trained using Kaggle's free TPU quota. 

[![Jupyter](https://img.shields.io/badge/Jupiter-View%20Notebook-orange?&logo=Jupyter)](https://nbviewer.jupyter.org/url/GabrieleSgroi.github.io/Notebooks/hubmap-segmentation%20%281%29.ipynb)
