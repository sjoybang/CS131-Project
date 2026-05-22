from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import numpy as np


def build_svm_pipeline(kernel='rbf', C=1.0, gamma='scale'):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42)),
    ])


def train(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_cv(pipeline, X, y, cv=5):
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')
    print(f"CV ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores
