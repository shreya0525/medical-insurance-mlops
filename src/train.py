import numpy as np
import pickle
import yaml
from sklearn.linear_model import LogisticRegression
import os

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def main():
    params = load_params()

    X = np.load("data/processed/X_train.npy")
    y = np.load("data/processed/y_train.npy")

    model = LogisticRegression(C=params["train"]["C"], random_state=params["train"]["random_state"])
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    with open("models/logreg_model.pkl", "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    main()
