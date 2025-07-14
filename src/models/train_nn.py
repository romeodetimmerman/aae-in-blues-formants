import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def load_data(csv_path="../../data/interim/formant_wide.csv"):
    """
    load and split formant feature data for neural network

    params
    ------
    csv_path: str, optional
        path to csv file with features

    returns
    -------
    X_train_flat: np.ndarray
        training features (flat, for mlp)
    X_test_flat: np.ndarray
        test features (flat, for mlp)
    X_train_seq: np.ndarray
        training features (seq, for cnn/rnn)
    X_test_seq: np.ndarray
        test features (seq, for cnn/rnn)
    y_train: np.ndarray
        training labels
    y_test: np.ndarray
        test labels
    """
    df = pd.read_csv(csv_path)

    # get all f1_XX and f2_XX columns
    feature_cols = [
        col for col in df.columns if col.startswith("f1_") or col.startswith("f2_")
    ]
    X = df[feature_cols].values

    y = df["aae_realization"].values

    # normalize features
    dtype = np.float32
    X = X.astype(dtype)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # reshape for cnn/rnn: (samples, time steps, channels)
    X_seq = X.reshape(-1, 200, 1)
    # keep 2d for mlp
    X_flat = X.reshape(-1, 200)

    y = y.astype(np.float32)

    # split for seq models
    X_train_seq, X_test_seq, y_train, y_test = train_test_split(
        X_seq, y, test_size=0.2, random_state=42, stratify=y
    )
    # split for mlp
    X_train_flat, X_test_flat, _, _ = train_test_split(
        X_flat, y, test_size=0.2, random_state=42, stratify=y
    )

    return (X_train_flat, X_test_flat, X_train_seq, X_test_seq, y_train, y_test)


def build_model(X_train, y_train, class_weight, model_type="mlp"):
    """
    build and train keras model

    params
    ------
    X_train: np.ndarray
        training features
    y_train: np.ndarray
        training labels
    class_weight: dict
        class weights for training
    model_type: str
        "mlp", "cnn", or "rnn"

    returns
    -------
    model: keras.Sequential
        trained keras model
    """
    if model_type == "mlp":
        # mlp: dense layers for flat input
        model = keras.Sequential(
            [
                layers.Input(shape=(X_train.shape[1],), name="input_layer"),
                layers.Dense(64, activation="relu", name="dense_1"),
                layers.Dropout(0.2, name="dropout_1"),
                layers.Dense(32, activation="relu", name="dense_2"),
                layers.Dense(1, activation="sigmoid", name="output_layer"),
            ]
        )
    elif model_type == "cnn":
        # cnn: 1d conv for sequential input
        model = keras.Sequential(
            [
                layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
                layers.Conv1D(8, 5, activation="relu"),
                layers.MaxPooling1D(2),
                layers.Flatten(),
                layers.Dropout(0.2),
                layers.Dense(16, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
    elif model_type == "rnn":
        # rnn: gru for sequential input
        model = keras.Sequential(
            [
                layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
                layers.GRU(8, return_sequences=False),
                layers.Dense(8, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["precision", "recall", "auc"],
    )

    # early stopping callback
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_auc", patience=3, restore_best_weights=True, mode="max"
    )

    # fit model
    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        class_weight=class_weight,
        verbose=2,
    )

    return model


def main():
    """
    train and save mlp, cnn, and rnn models
    """
    X_train_flat, X_test_flat, X_train_seq, X_test_seq, y_train, y_test = load_data()

    # compute class weights
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    # soften class weights
    alpha = 0.05
    class_weight = {k: 1.0 + alpha * (v - 1.0) for k, v in class_weight.items()}

    # train and save mlp
    model_mlp = build_model(X_train_flat, y_train, class_weight, model_type="mlp")
    model_mlp.save("../../models/neural_network_mlp.keras")

    # train and save cnn
    model_cnn = build_model(X_train_seq, y_train, class_weight, model_type="cnn")
    model_cnn.save("../../models/neural_network_cnn.keras")

    # train and save rnn
    model_rnn = build_model(X_train_seq, y_train, class_weight, model_type="rnn")
    model_rnn.save("../../models/neural_network_rnn.keras")


if __name__ == "__main__":
    main()
