# Machine-Learning-Optimization-XGBoost-Hyperparameter-Tuning-with-Optuna-for-Digit-Classification
Machine Learning Optimization: XGBoost Hyperparameter Tuning with Optuna for Digit Classification


This example demonstrates using Optuna to optimize an XGBoost binary classifier 
for a challenging computer vision task using the classic digits dataset. The 
project showcases both hyperparameter tuning and model selection optimization 
in a real-world scenario. 

The digits dataset contains 8x8 grayscale images of 
handwritten numbers (0-9). We create an intentionally challenging 
classification problem by: 

- Selecting only digits 3-9 to create class imbalance
- Converting it to a binary task: classifying digits as either "high" (6-9) or 
"medium" (3-5)

The optimization process explores multiple XGBoost architectures (gbtree, 
gblinear, and dart boosters) while simultaneously tuning their hyperparameters. 
Key optimized parameters include:

- Tree structure (depth, child weights, growth policies)
- Regularization (L1/L2)
- Sampling strategies
- Learning rates
- DART-specific parameters when applicable

This example demonstrates how Optuna can efficiently navigate a complex, 
conditional parameter space where some hyperparameters only apply to certain 
model types. The optimization aims to maximize classification accuracy while 
handling the inherent challenges of computer vision data.



# Installation

### Clone this repository:

git clone https://github.com/yourusername/xgboost-optuna-optimization.git

cd xgboost-optuna-optimization

### Install the required packages:

pip install numpy optuna scikit-learn xgboost

# Usage

### Run the optimization script:

python optimize_xgboost.py

# The script will:

##### Load and preprocess the digits dataset
##### Run 100 optimization trials (or until the 600-second timeout)
##### Print the best hyperparameters and their corresponding accuracy
