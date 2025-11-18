import pandas as pd
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
)
from joblib import dump


def train_model(path: str):
    df = pd.read_csv(path)

    # Use the new features as well
    feature_cols = ["pd_diff", "pf_diff", "pa_diff", "rest_diff", "b2b_diff"]
    target_col = "home_win"

    # Train on older seasons, test on more recent ones
    train_df = df[df["Season"] <= 2020]
    test_df = df[df["Season"] > 2020]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # Base model: gradient boosting (usually better than RF for tabular data)
    base_model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=400,
        random_state=42,
    )

    # Calibrate probabilities for better realism
    model = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba > 0.5).astype(int)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("ROC AUC:", roc_auc_score(y_test, proba))
    print("Brier score:", brier_score_loss(y_test, proba))
    print("Log loss:", log_loss(y_test, proba))

    # Save model and feature names
    Path("models").mkdir(exist_ok=True)
    dump(model, "models/game_model.joblib")
    dump(feature_cols, "models/feature_cols.joblib")
    print("Saved calibrated model to models/game_model.joblib")


def main():
    dataset_path = "data/processed/model_dataset.csv"
    train_model(dataset_path)


if __name__ == "__main__":
    main()
