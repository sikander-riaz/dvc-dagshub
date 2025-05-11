import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Create directory for artifacts if it doesn't exist
os.makedirs("artifacts", exist_ok=True)

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=43
)
max_depth = 5
n_estimators = 1000  # Note: This isn't used for Decision Trees

# mlflow.set_experiment('iris-dt')
with mlflow.start_run(experiment_id="251274238044326048"):
    # Model training
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log parameters and metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("max_depth", max_depth)

    # Log model as artifact
    mlflow.sklearn.log_model(model, "trained_model")

    # Create and log confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("artifacts/confusion_matrix.png")
    mlflow.log_artifact("artifacts/confusion_matrix.png")
    plt.close()

    # Log feature importance
    feature_importances = pd.DataFrame(
        {"feature": iris.feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    feature_importances.to_csv("artifacts/feature_importances.csv", index=False)
    mlflow.log_artifact("artifacts/feature_importances.csv")

    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=feature_importances)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("artifacts/feature_importance_plot.png")
    mlflow.log_artifact("artifacts/feature_importance_plot.png")
    plt.close()

    # Log classification report
    report = classification_report(
        y_test, y_pred, target_names=iris.target_names, output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv("artifacts/classification_report.csv")
    mlflow.log_artifact("artifacts/classification_report.csv")

    # Visualize the decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(
        model,
        filled=True,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        rounded=True,
    )
    plt.title("Decision Tree Visualization")
    plt.savefig("artifacts/decision_tree.png", dpi=200, bbox_inches="tight")
    mlflow.log_artifact("artifacts/decision_tree.png")
    plt.close()
    mlflow.log_artifact(__file__)
    # mlflow.sklearn.log_model(dt,'decision tree')

    # Log actual vs predicted values
    results_df = pd.DataFrame(
        {"actual": y_test, "predicted": y_pred, "correct": y_test == y_pred}
    )
    results_df["actual_class"] = results_df["actual"].map(
        {i: name for i, name in enumerate(iris.target_names)}
    )
    results_df["predicted_class"] = results_df["predicted"].map(
        {i: name for i, name in enumerate(iris.target_names)}
    )
    results_df.to_csv("artifacts/prediction_results.csv", index=False)
    mlflow.log_artifact("artifacts/prediction_results.csv")
