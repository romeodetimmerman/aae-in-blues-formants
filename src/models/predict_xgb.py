import xgboost as xgb
from train_xgb import load_data
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
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


def load_model():
    """
    load gradient boosting model from file

    returns
    -------
    model: xgb.XGBClassifier
        trained xgboost classifier
    """
    model = xgb.XGBClassifier()
    model.load_model("../../models/gradient_boost.pkl")
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


def confusion_plot(y_test, y_pred):
    """
    plot confusion matrix and save to file

    params
    ------
    y_test: array-like
        true labels
    y_pred: array-like
        predicted labels
    """
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
    plt.title("Confusion matrix for XGB")
    plt.xlabel("Predicted values")
    plt.ylabel("Actual values")
    plt.tight_layout()
    plt.savefig("../../figures/confusion_matrix_xgb.png", dpi=600)
    plt.close()


def shap_plot(model, X_train, train_vowel_ids):
    """
    plot shap values and save summary/bar/waterfall plots

    params
    ------
    model: xgb.XGBClassifier
        trained model
    X_train: pd.DataFrame or np.ndarray
        training features
    train_vowel_ids: pd.Series
        ids for each training sample
    """

    # custom colors
    colors = ["#E1812C", "#3373A1"]
    custom_neg_color = colors[0]
    custom_pos_color = colors[1]
    default_pos_color = "#ff0051"
    default_neg_color = "#008bfb"

    # create a copy of X_train with formatted column names for SHAP plotting
    if hasattr(X_train, "columns"):
        formatted_columns = [
            col.replace("_", " ").capitalize() for col in X_train.columns
        ]
        X_train_pretty = X_train.copy()
        X_train_pretty.columns = formatted_columns
    else:
        X_train_pretty = X_train

    explainer = shap.Explainer(model, X_train_pretty)
    shap_values = explainer(X_train_pretty)

    # beeswarm plot with custom colormap
    shap.summary_plot(
        shap_values,
        X_train_pretty,
        max_display=15,
        show=False,
        cmap="Blues",
    )
    plt.savefig("../../figures/shap_summary_beeswarm.png", dpi=600, bbox_inches="tight")
    plt.close()

    # bar plot (all bars in custom blue)
    fig, ax = plt.subplots()
    shap.summary_plot(
        shap_values,
        X_train_pretty,
        plot_type="bar",
        max_display=15,
        show=False,
    )

    # set all bars to custom blue
    custom_blue = "#3373A1"
    for bar in ax.patches:
        bar.set_facecolor(custom_blue)

    plt.xlabel("Mean absolute SHAP value (impact on model output)")
    plt.savefig("../../figures/shap_summary_bar.png", dpi=600, bbox_inches="tight")
    plt.close()

    # waterfall plots for selected vowels
    vowels_of_interest = ["001-001-014", "001-001-020"]
    idxs_of_interest_in_X_train = []
    train_vowel_ids_reset = train_vowel_ids.reset_index(drop=True)

    for vowel_id in vowels_of_interest:
        idxs = train_vowel_ids_reset[train_vowel_ids_reset == vowel_id].index.tolist()
        if idxs:
            idxs_of_interest_in_X_train.extend(idxs)
            for idx in idxs:
                print("\nindex in X_train for vowel_id {}: {}".format(vowel_id, idx))
                print(
                    "verifying vowel id at that index:", train_vowel_ids_reset.iloc[idx]
                )
        else:
            print("\nvowel_id {} not found in train_vowel_ids".format(vowel_id))

    for vowel in idxs_of_interest_in_X_train:
        shap.plots.waterfall(
            shap_values[vowel],
            max_display=15,
            show=False,
        )
        # manually update colors for waterfall plot
        for fc in plt.gcf().get_children():
            for fcc in fc.get_children():
                if isinstance(fcc, matplotlib.patches.FancyArrow):
                    if (
                        matplotlib.colors.to_hex(fcc.get_facecolor())
                        == default_pos_color
                    ):
                        fcc.set_facecolor(custom_pos_color)
                        fcc.set_edgecolor(custom_pos_color)
                    elif (
                        matplotlib.colors.to_hex(fcc.get_facecolor())
                        == default_neg_color
                    ):
                        fcc.set_facecolor(custom_neg_color)
                        fcc.set_edgecolor(custom_neg_color)
                elif isinstance(fcc, plt.Text):
                    if matplotlib.colors.to_hex(fcc.get_color()) == default_pos_color:
                        fcc.set_color(custom_pos_color)
                    elif matplotlib.colors.to_hex(fcc.get_color()) == default_neg_color:
                        fcc.set_color(custom_neg_color)
        plt.savefig(
            "../../figures/shap_waterfall_{}.png".format(
                train_vowel_ids_reset.iloc[vowel]
            ),
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()


def plot_precision_recall_vs_threshold(model, X_test, y_test):
    """
    plot precision and recall as a function of threshold for both classes

    params
    ------
    model: xgb.XGBClassifier
        trained xgboost model
    X_test: np.ndarray or pd.DataFrame
        test features
    y_test: np.ndarray or pd.Series
        test labels
    """
    # class 1 (default)
    y_scores_1 = model.predict_proba(X_test)[:, 1]
    precision_1, recall_1, thresholds_1 = precision_recall_curve(y_test, y_scores_1)

    # class 0 (invert labels and use prob for class 0)
    y_scores_0 = model.predict_proba(X_test)[:, 0]
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
    plt.title("precision–recall vs threshold (both classes) for XGB")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("../../figures/precision_recall_vs_threshold_xgb.png", dpi=600)
    plt.close()


def plot_roc_curve(model, X_test, y_test):
    """
    plot roc curve and display auc

    params
    ------
    model: xgb.XGBClassifier
        trained xgboost model
    X_test: np.ndarray or pd.DataFrame
        test features
    y_test: np.ndarray or pd.Series
        test labels
    """
    y_scores = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    auc = roc_auc_score(y_test, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label="ROC curve (AUC = {:.2f})".format(auc))
    plt.plot([0, 1], [0, 1], "k--", label="random guess")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("ROC curve (monophthongs) for XGB")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("../../figures/roc_curve_xgb.png", dpi=600)
    plt.close()


def main():
    """
    run model prediction, evaluation, and visualization
    """
    X_train, X_test, y_train, y_test, train_vowel_ids, test_vowel_ids = load_data()
    model = load_model()

    # use threshold of 0.6 for prediction
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob > 0.6).astype(int)

    # plot precision-recall vs threshold
    plot_precision_recall_vs_threshold(model, X_test, y_test)

    # plot roc curve
    plot_roc_curve(model, X_test, y_test)

    # evaluate model
    results = evaluate_model(y_test, y_pred, y_pred_prob)
    print("\nmodel results:")
    print(results)

    # confusion matrix
    confusion_plot(y_test, y_pred)

    print("\nclassification report:")
    print(classification_report(y_test, y_pred))

    # shap plots
    shap_plot(model, X_train, train_vowel_ids)
    print("\ndone")


if __name__ == "__main__":
    main()
