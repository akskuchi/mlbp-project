from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV

import load_data

X, y = load_data()

X_train, X_test, y_train, y_test = train_test_split(X, y)

classifier = Pipeline([
    ('std', StandardScaler()),
    ('pca', PCA(n_components=100)),
    ('lrcv', LogisticRegressionCV(
        Cs=[0.1, 0.2, 0.5, 0.8, 1, 2, 5]))
])

classifier.fit(X_train, y_train)
accuracy = classifier.score(X_test, y_test)

print(accuracy)
