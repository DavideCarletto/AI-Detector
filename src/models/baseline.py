
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

def train_baseline(X_train, Y_train, model, max_feats = 50000,):

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=max_feats)), # Tokenizer
        ('clf', model) # Classifier
    ])

    pipe.fit(X_train, Y_train) # Train the entire pipeline

    return pipe