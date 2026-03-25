from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate(pipe, X_test, Y_test):
    
    Y_pred = pipe.predict(X_test) 
    
    Y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(Y_test, Y_pred),
        "roc_auc": roc_auc_score(Y_test, Y_proba)
    }
    
    plt.figure(figsize=(8, 6))
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    fig = plt.gcf()
    plt.close()
    
    return metrics, fig