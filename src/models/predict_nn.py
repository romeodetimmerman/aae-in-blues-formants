from tensorflow import keras
from train_nn import load_data
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    precision_recall_curve,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils import plot_model


def load_model(model_type):
    """
    load neural network model from file

    params
    ------
    model_type: str
        "mlp", "cnn", or "rnn"

    returns
    -------
    model: keras.Model
        trained keras model
    """
    model_paths = {
        "mlp": "../../models/neural_network_mlp.keras",
        "cnn": "../../models/neural_network_cnn.keras",
        "rnn": "../../models/neural_network_rnn.keras",
    }
    model = keras.models.load_model(model_paths[model_type])
    return model


def evaluate_model(y_test, y_pred, y_pred_prob):
    """
    compute evaluation metrics for predictions

    params
    ------
    y_test: array-like
        true labels
    y_pred: array-like
        predicted labels
    y_pred_prob: array-like
        predicted probabilities

    returns
    -------
    results: dict
        accuracy, precision, recall, f1, roc_auc
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }
    return results


def confusion_plot(y_test, y_pred, model_type=None):
    """
    plot confusion matrix and save to file

    params
    ------
    y_test: array-like
        true labels
    y_pred: array-like
        predicted labels
    model_type: str or None
        model type for filename ("mlp", "cnn", "rnn")

    returns
    -------
    None
    """
    # plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    class_names = ["Diphthong", "Monophthong"]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion matrix for {model_type.upper()}")
    plt.xlabel("Predicted values")
    plt.ylabel("Actual values")
    plt.tight_layout()
    plt.savefig(f"../../figures/confusion_matrix_{model_type}.png", dpi=600)
    plt.close()


def plot_precision_recall_vs_threshold(model, X_test, y_test, model_type):
    """
    plot precision and recall as a function of threshold for both classes

    params
    ------
    model: keras.Model
        trained keras model
    X_test: np.ndarray
        test features
    y_test: np.ndarray
        test labels

    returns
    -------
    None
    """
    # get predicted probabilities for class 1
    y_scores_1 = model.predict(X_test).ravel()
    precision_1, recall_1, thresholds_1 = precision_recall_curve(y_test, y_scores_1)

    # get predicted probabilities for class 0 (1 - prob for class 1)
    y_scores_0 = 1 - y_scores_1
    precision_0, recall_0, thresholds_0 = precision_recall_curve(1 - y_test, y_scores_0)

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds_1, precision_1[:-1], label="precision (class 1)")
    plt.plot(thresholds_1, recall_1[:-1], label="recall (class 1)")
    plt.plot(
        thresholds_0, precision_0[:-1], label="precision (class 0)", linestyle="--"
    )
    plt.plot(thresholds_0, recall_0[:-1], label="recall (class 0)", linestyle="--")
    plt.xlabel("threshold")
    plt.ylabel("score")
    plt.title(f"precision–recall vs threshold (both classes) for {model_type.upper()}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(
        f"../../figures/precision_recall_vs_threshold_{model_type}.png", dpi=600
    )
    plt.close()


def plot_roc_curve(model, X_test, y_test, model_type):
    """
    plot roc curve and display auc

    params
    ------
    model: keras.Model
        trained keras model
    X_test: np.ndarray
        test features
    y_test: np.ndarray
        test labels

    returns
    -------
    None
    """
    y_scores = model.predict(X_test).ravel()
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    auc = roc_auc_score(y_test, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="random guess")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title(f"ROC curve (monophthongs) for {model_type.upper()}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"../../figures/roc_curve_{model_type}.png", dpi=600)
    plt.close()


def main():
    """
    run model prediction, evaluation, and visualization for all model types

    returns
    -------
    None
    """
    X_train_flat, X_test_flat, X_train_seq, X_test_seq, y_train, y_test = load_data()
    model_types = ["mlp", "cnn", "rnn"]
    X_tests = {"mlp": X_test_flat, "cnn": X_test_seq, "rnn": X_test_seq}

    for model_type in model_types:
        print(f"\nevaluating {model_type.upper()} model")
        model = load_model(model_type)
        X_test = X_tests[model_type]
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.6).astype(int).flatten()

        # plot precision-recall vs threshold
        plot_precision_recall_vs_threshold(model, X_test, y_test, model_type)

        # plot roc curve
        plot_roc_curve(model, X_test, y_test, model_type)

        # evaluate model
        results = evaluate_model(y_test, y_pred, y_pred_prob)
        print("\nmodel results:")
        print(results)

        # confusion matrix
        confusion_plot(y_test, y_pred, model_type=model_type)

        print("\nclassification report:")
        print(classification_report(y_test, y_pred))

        # plot model architecture
        plot_model(
            model,
            to_file=f"../../figures/model_architecture_{model_type}.png",
            rankdir="LR",
            show_shapes=True,
            show_layer_names=True,
            show_layer_activations=True,
            dpi=600,
        )


if __name__ == "__main__":
    main()
