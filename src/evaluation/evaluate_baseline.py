import pandas as pd
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

from src.utils.config_loader import cfg

def evaluate():

    print("Evaluating model...")

    models_path = cfg["paths"]["models"]
    data_path = cfg["paths"]["data_processed"]
    reports_path = "reports/figures"
    os.makedirs(reports_path, exist_ok=True)

    df = pd.read_csv(data_path)
    _, X_test, _, Y_test = train_test_split(
        df['text'].values, df['generated'].values, 
        test_size=0.2, random_state=42, stratify=df['generated'].values
    )

    vectorizer = joblib.load(os.path.join(models_path, "tfidf_vectorizer.joblib"))
    model = joblib.load(os.path.join(models_path, "baseline_model.joblib"))

    tfidf_test_vectors = vectorizer.transform(X_test)

    Y_pred = model.predict(tfidf_test_vectors)
    Y_proba = model.predict_proba(tfidf_test_vectors)[:, 1]

    
    metrics = {
        "accuracy": accuracy_score(Y_test, Y_pred),
        "roc_auc": roc_auc_score(Y_test, Y_proba)
    }

    print("Classification report:")
    print(classification_report(Y_test, Y_pred))
    print(metrics)
    
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    plt.figure(figsize=(8, 6))
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.savefig(os.path.join(reports_path, "confusion_matrix.png"))
    
    print("Evaluation complete. Metrics and figures saved in reports/")

if __name__ == "__main__":
    evaluate()