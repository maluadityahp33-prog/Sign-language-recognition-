"""Step 2 - train and compare classifiers on the collected landmarks.

Run:  python src/train_model.py
"""
import matplotlib

matplotlib.use("Agg")  # save plots to a file, no window needed
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config


def mirror(X):
    """Flip x-coordinates so the model also works with the other hand."""
    flipped = X.copy().reshape(len(X), -1, 3)
    flipped[:, :, 0] *= -1
    return flipped.reshape(len(X), -1)


def main():
    if not config.DATA_FILE.exists():
        raise SystemExit("No data found. Run  python src/collect_data.py  first.")

    df = pd.read_csv(config.DATA_FILE)
    X = df.drop(columns="label").to_numpy(dtype=np.float32)
    y = df["label"].astype(str).to_numpy()

    counts = pd.Series(y).value_counts().sort_index()
    print("Samples per sign:\n", counts.to_string(), "\n")
    if len(counts) < 2:
        raise SystemExit("Need at least 2 different signs to train.")
    if counts.min() < 5:
        raise SystemExit("Some signs have fewer than 5 samples. Collect more data.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    # Augment only the training split with mirrored (other-hand) copies
    X_aug = np.vstack([X_train, mirror(X_train)])
    y_aug = np.concatenate([y_train, y_train])

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "SVM (RBF)": make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(SVC(C=10), cv=3, ensemble=False),  # gives probabilities
        ),
        "Neural Net (MLP)": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_aug, y_aug)
        acc = accuracy_score(y_test, model.predict(X_test))
        results[name] = acc
        print(f"{name:<18} test accuracy: {acc:.3f}")

    best_name = max(results, key=results.get)
    best_model = models[best_name]
    print(f"\nBest model: {best_name} ({results[best_name]:.3f})\n")

    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, ax=ax, cmap="Blues", colorbar=False, xticks_rotation=90
    )
    ax.set_title(f"Confusion matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(config.CONFUSION_MATRIX_FILE, dpi=150)
    plt.close(fig)

    # Refit the best model on ALL data (plus mirrored copies) for the final app
    final = clone(best_model)
    final.fit(np.vstack([X, mirror(X)]), np.concatenate([y, y]))
    joblib.dump(
        {"model": final, "name": best_name, "test_accuracy": results[best_name]},
        config.MODEL_FILE,
    )
    print(f"Model saved to:            {config.MODEL_FILE}")
    print(f"Confusion matrix saved to: {config.CONFUSION_MATRIX_FILE}")


if __name__ == "__main__":
    main()
