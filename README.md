# IPL-Predictive-Analysis

This algorithm uses a neural network approach to predict IPL outcomes, specifically if a batsman will get out based on input constrainsts. It yeilds a test accuracy of 0.95.

First the data is preprocessed for the network and is encoded using scitkitlearns's LabelEncoder. Next, a MultiLayer Preceptron model is used with 4 hidden layers (Sigmoid x 3, ReLU) and an ouput layer (Linear). Finally, Training and Testing Accuracy is done, and their metrics are displayed.
