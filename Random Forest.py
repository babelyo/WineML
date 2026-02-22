import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import joblib


def quality_group(q):
    if q <= 5:
        return "low"
    elif q <= 7:
        return "medium"
    else:
        return "high"


df = pd.read_csv("winequality_white.csv", sep=';')

X = df.iloc[:, :-1]  # wszystkie cechy
y = df.quality
y_grouped = y.apply(quality_group)

X_train, X_test, y_train, y_test = train_test_split(X, y_grouped, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=8, max_features=8, min_samples_split=8, random_state=42,
                               class_weight="balanced")
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))

# zapis i załadowanie modelu

joblib.dump(model, 'moj_tree_classifier.model')  #zapis modelu do pliku
model_new = joblib.load('moj_tree_classifier.model')  # wczytanie modelu

#testowanie modelu
print(X_test)
print(X_test.iloc[4:10, :])  # wybieram jeden wiersz
print(model_new.predict(X_test.iloc[4:10, :]))  # liczę
