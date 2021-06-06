# Portfolio
## Deep learning projects: Supervised learning 
### Vehicle motion prediction with 3d CNN 
I have trained a 3d convolutional neural network to predict the future trajectories of vehicles as an entry for the [Lyft Motion Prediction for Autonomous Vehicles](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles) competition on Kaggle. The model takes as input few frames of bird's-eye view images containing the visual representation of all the vehicles in the scene and, through intermediate layers including also 3d convolutions (1 temporal+ 2 spatial dimensions), predicts 3 possible future trajectories for the target vehicle and associate a probability to each of them. The model easily outperforms the baseline benchmark set by the competition after being trained on approximately 10% of the training set. 

[![Jupyter](https://img.shields.io/badge/Jupiter-View%20Notebook-orange?&logo=Jupyter)](https://nbviewer.jupyter.org/url/GabrieleSgroi.github.io/Notebooks/lyft-vehicles-motion-prediction-3d-cnn-with-keras.ipynb)

### Image segmentation to identify glomeruli in Kidney

I trained a [U-Net](https://arxiv.org/pdf/1505.04597.pdf) like architecture to predict segmentation masks in order to identify glomeruli inspired by the task of the [HuBMAP - Hacking the Kidney](https://www.kaggle.com/c/hubmap-kidney-segmentation) competition on Kaggle. The model adds an attention mechanism to the U-Net through the use of the [Convolutional Block Attention Module](https://arxiv.org/pdf/1807.06521.pdf).

[![Jupyter](https://img.shields.io/badge/Jupiter-View%20Notebook-orange?&logo=Jupyter)](https://nbviewer.jupyter.org/url/GabrieleSgroi.github.io/Notebooks/hubmap-segmentation%20%281%29.ipynb)
