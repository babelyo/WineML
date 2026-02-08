import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import joblib

from sklearn.model_selection import GridSearchCV


def quality_group(q):
    if q <= 5:
        return "low"
    elif q <= 7:
        return "medium"
    else:
        return "high"


df = pd.read_csv('winequality_white.csv', sep=';')
print(df)
print(df.quality.value_counts())

X = df.iloc[:,:-1]  # wszystkie kolumny, bez ostatniej
y = df.quality  # kolumna quality
y_grouped = y.apply(quality_group)

X_train, X_test, y_train, y_test = train_test_split(X, y_grouped, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(criterion= 'gini', max_depth= 8, max_features= 10, min_samples_split= 5, random_state=42)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))
print(pd.DataFrame(model.feature_importances_, X.columns))  #które cechy najwazniejsze
print(classification_report(y_test, model.predict(X_test)))
joblib.dump(model, 'moj_tree_classifier.model')

# Szukanie nalepszych parametrów dla drzewa decyzyjnego, zględem najlepszego dopasowania.
'''
print(f'liczba cech {X.shape[1]}')
model = DecisionTreeClassifier()
params = {
    'max_depth': range(2, 10),
    'max_features': range(2, X.shape[1]+1),
    'min_samples_split': range(2, 10),
    'criterion': ['gini', 'entropy', 'log_loss']
}
grid = GridSearchCV(model, params, scoring='accuracy', cv=5, verbose=1)
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_estimator_)
print(grid.best_score_)
'''