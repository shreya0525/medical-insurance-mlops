import pandas as pd
import numpy as np
import yaml
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def main():
    params = load_params()
    threshold = params["preprocess"]["target_threshold"]

    df = pd.read_csv("data/raw/insurance.csv")
    df['target'] = np.where(df['charges'] > threshold, 1, 0)

    cat_cols = ['sex', 'smoker', 'region']
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    X = df.drop(['charges', 'target'], axis=1)
    y = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/X_train.npy", X_scaled)
    np.save("data/processed/y_train.npy", y)

if __name__ == "__main__":
    main()
