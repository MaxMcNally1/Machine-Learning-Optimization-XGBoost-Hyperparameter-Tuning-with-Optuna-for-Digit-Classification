"""
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

"""

import numpy as np
import optuna

import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb


def objective(trial):
    # Load the digits dataset and create a challenging binary classification
    digits = sklearn.datasets.load_digits()
    # Only take digits 3-9, then classify as high (6-9) vs medium (3-5)
    mask = (digits.target >= 3) & (digits.target <= 9)
    data = digits.data[mask]
    target = (digits.target[mask] >= 6).astype(int)

    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dvalid = xgb.DMatrix(valid_x, label=valid_y)

    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        # use exact for small dataset.
        "tree_method": "exact",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    bst = xgb.train(param, dtrain)
    preds = bst.predict(dvalid)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)
    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
