import numpy as np
import pandas as pd
import os 
import joblib

from src.utils.config_loader import cfg

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

import seaborn as sns

processed_data_path = cfg["paths"]["data_processed"]
models_output_path = cfg["paths"]["models"]
vectorizer_file_name = "tfidf_vectorizer.joblib"
baseline_file_name = "baseline_model.joblib"

def train_baseline():

    print("Loading data...")
    df = pd.read_csv(processed_data_path)

    print("Data Loaded! Now splitting...")
    X_train, _, Y_train, _ = train_test_split(df['text'].values, df['generated'].values, test_size=0.2, random_state=42, stratify=df['generated'].values )


    print("Data splitted! Now training...")
    # Tokenization
    tfidf_vectorizer = TfidfVectorizer(max_features=50000)
    tfidf_train_vectors = tfidf_vectorizer.fit_transform(X_train)

    # Training classifier
    classifier = SGDClassifier(loss='log_loss')
    classifier.fit(tfidf_train_vectors, Y_train)

    print("Training Done! Saving models...")

    os.makedirs(models_output_path, exist_ok=True)

    vectorizer_path = os.path.join(models_output_path, vectorizer_file_name)
    baseline_path = os.path.join(models_output_path, baseline_file_name)

    joblib.dump(tfidf_vectorizer ,vectorizer_path, compress=3)
    joblib.dump(classifier, baseline_path, compress=3)

    print(f"Models saved successfully in: {models_output_path}")


if __name__ == "__main__":
    train_baseline()