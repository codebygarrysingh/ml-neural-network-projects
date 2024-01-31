# ml-neural-network-projects

## LSTM Neural Network - Room Occupancy Estimator

This Room Occupancy Estimator is a comprehensive data science project that involves multiple stages, from exploratory data analysis (EDA) to neural network training and evaluation. The goal is to predict room occupancy based on various sensor readings over time. The implementation is presented in a Python notebook using popular libraries such as pandas, seaborn, matplotlib, and PyTorch.

### Exploratory Data Analysis (EDA):
The initial step involves exploring the dataset ('Occupancy_Estimation.csv') to understand the distribution of sensor readings over time and their correlation with room occupancy. The EDA utilizes line plots to visualize the temporal patterns of sensor readings and room occupancy.

### Handling Missing Values and Train/Test Split:
The notebook addresses missing values in the dataset using imputation techniques, ensuring a clean dataset for further analysis. Additionally, a meaningful train/test split is established to evaluate the performance of the machine learning models.

### Neural Network Design:
The notebook introduces a Long Short-Term Memory (LSTM) neural network, acknowledging the temporal nature of the sensor data. The model architecture is defined using PyTorch, with a single LSTM layer followed by a fully connected layer for prediction.

### Training the Neural Network:
The neural network is trained using the training set. The notebook incorporates data normalization using a StandardScaler and utilizes Mean Squared Error (MSE) as the loss function. The training process involves optimization using the Adam optimizer, iterating through the dataset in batches.

### Evaluating Model Performance:
The notebook evaluates the trained model's performance on the test set. It computes and reports metrics such as Mean Squared Error (MSE) and R-squared (R2) score. Training and test losses are visualized over epochs to provide insights into the model's learning process.

### Final Visualization:
The training and testing losses are visualized over epochs using a line plot. This graphical representation helps in understanding how the model's performance evolves during training.

This Room Occupancy Estimator serves as a valuable resource for predicting room occupancy based on sensor readings, showcasing the integration of data exploration, preprocessing, neural network design, training, and evaluation. Users can adapt and extend this notebook for similar tasks in their own projects.
