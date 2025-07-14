import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


def load_data(csv_path="../../data/processed/formant_features.csv"):
    """
    load and split formant feature data

    returns
    -------
    X_train: pd.DataFrame
        training features
    X_test: pd.DataFrame
        test features
    y_train: pd.Series
        training labels
    y_test: pd.Series
        test labels
    train_vowel_ids: pd.Series
        vowel ids for training set
    test_vowel_ids: pd.Series
        vowel ids for test set
    """
    df = pd.read_csv(csv_path)

    # map perceptive label to 0 and 1
    df["perceptive_label"] = df["perceptive_label"].map(
        {"monophthong": 1, "diphthong": 0}
    )

    # select features
    vowel_ids = df["vowel_id"]  # keep for reference later

    X = df.drop(columns=["perceptive_label", "vowel_id"])
    y = df["perceptive_label"]

    # train test split
    X_train, X_test, y_train, y_test, train_vowel_ids, test_vowel_ids = (
        train_test_split(X, y, vowel_ids, test_size=0.2, random_state=42, stratify=y)
    )

    return X_train, X_test, y_train, y_test, train_vowel_ids, test_vowel_ids


def build_model(X_train, y_train):
    """
    fit gradient boosting model

    params
    ------
    X_train: pd.DataFrame
        training features
    y_train: pd.Series
        training labels

    returns
    -------
    model: xgb.XGBClassifier
        trained xgboost classifier
    """
    # class weights
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

    # fit model
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )

    model.fit(X_train, y_train)

    return model


def main():
    """
    train and save gradient boosting model
    """
    X_train, X_test, y_train, y_test, train_vowel_ids, test_vowel_ids = load_data()
    model = build_model(X_train, y_train)
    model.save_model("../../models/gradient_boost.pkl")


if __name__ == "__main__":
    main()
