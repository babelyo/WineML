import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

df = pd.read_csv('winequality_white.csv', sep=';')
print(df)

print(df.quality.value_counts())

X = df.iloc[:,:-1]  # wszystkie kolumny, bez ostatniej
y = df.quality  # kolumna target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(criterion= 'gini', max_depth= 8, max_features= 8, min_samples_split= 8, random_state=42)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))
print(pd.DataFrame(model.feature_importances_, X.columns))  #które cechy najwazniejsze
print(classification_report(y_test, model.predict(X_test)))
print(f'liczba cech {X.shape[1]}')
# Szukanie nalepszych parametrów dla drzewa decyzyjnego, zględem najlepszego dopasowania.
'''
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