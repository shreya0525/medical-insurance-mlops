import numpy as np
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix

def main():
    X = np.load("data/processed/X_train.npy")
    y = np.load("data/processed/y_train.npy")

    with open("models/logreg_model.pkl", "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)

    print(f"Accuracy: {acc}")
    print("Confusion Matrix:")
    print(cm)

    with open("metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Confusion Matrix:\n{cm}")

if __name__ == "__main__":
    main()
