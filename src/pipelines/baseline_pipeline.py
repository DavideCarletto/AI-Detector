import mlflow
import dagshub
import pandas as pd 
import os 
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

from src.utils.config_loader import cfg
from src.models.baseline import train_baseline
from src.evaluate import evaluate

processed_data_path = cfg["paths"]["data_processed"]
models_output_path = cfg["paths"]["models"]
vectorizer_file_name = "tfidf_vectorizer.joblib"
baseline_file_name = "baseline_model.joblib"

def run_pipeline(pipeline="baseline", run_name = "Baseline_SGD"):
    
    dagshub.init(repo_owner='carlettodavide', repo_name='AI-Detector', mlflow=True)

    if pipeline == "baseline":
        mlflow.set_experiment("Baseline Experiments")

        with mlflow.start_run(run_name=run_name):
                print("Loading data...")
                df = pd.read_csv(processed_data_path)

                print("Data Loaded! Now splitting...")
                X_train, X_test, Y_train, Y_test = train_test_split(df['text'].values, df['generated'].values, test_size=0.2, random_state=42, stratify=df['generated'].values )

                print("Data splitted! Now training...")

                # Tokenization
                max_feats = 50000
                loss = "log_loss"
                model= SGDClassifier(loss=loss, random_state=42)

                mlflow.log_param("max_features", max_feats)
                mlflow.log_param("model_type", "SGDClassifier")
                
                pipeline = train_baseline(X_train, Y_train, model, max_feats=max_feats)

                print("Training Done! Now evaluating...")

                metrics, fig = evaluate(pipeline, X_test, Y_test)

                # Log Metrics
                mlflow.log_metrics(metrics)
                mlflow.log_figure(fig, "plots/confusion_matrix.png")
            
                print("Evaluation Done! Saving models...")

                os.makedirs(models_output_path, exist_ok=True)
                pipeline_path = os.path.join(models_output_path, "full_pipeline.joblib")
                joblib.dump(pipeline, pipeline_path)

                print(f"Models saved successfully in: {models_output_path}")

                mlflow.sklearn.log_model(pipeline, "Detector")

                print("Pipeline completed! Final metrics: ", metrics)

if __name__ == "__main__":
    run_pipeline()