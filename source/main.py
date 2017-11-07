import numpy as np
import pandas as pd
from sklearn import preprocessing, linear_model, pipeline,\
    model_selection, decomposition

import os

resource_path = os.path.join(
    os.path.abspath('..'),
    'resources')

paths = [os.path.join(resource_path, p) for p in
         ['train_data.csv', 'train_labels.csv', 'test_data.csv']]

X, y, test_data = [pd.read_csv(p, header=None)
                   for p in paths]

X = X.values
y = y.values.ravel()

pl = pipeline.Pipeline(
    [('scaler', preprocessing.StandardScaler()),
     # at the moment the pipeline performs better without
     # PCA but is left here for reference. For optimal performance,
     # comment the PCA line before running the code.
     ('PCA', decomposition.PCA(n_components=0.8, svd_solver='full')),
     ('classifier', linear_model.LogisticRegression(random_state=0))])

scores = model_selection.cross_val_score(
    estimator=pl, X=X, y=y, cv=5, n_jobs=-1)

print('cross validation scores: {}'.format(scores))
print('cross validation accuracy: {} +/- {}'
      .format(np.mean(scores), np.std(scores)))

pl.fit(X, y)

pred_accuracy = pd.DataFrame(pl.predict(test_data))
pred_logloss = pd.DataFrame(pl.predict_proba(test_data))

out_path = os.path.join(
    os.path.abspath('..'),
    'output')

if not os.path.exists(out_path):
    os.makedirs(out_path)

acc_path = os.path.join(out_path, 'accuracy.csv')
logloss_path = os.path.join(out_path, 'logloss.csv')

pred_accuracy.index += 1
pred_logloss.index += 1

pred_accuracy.to_csv(acc_path, header=['Sample_label'],
                     index_label='Sample_id')
pred_logloss.to_csv(logloss_path,
                    header=['Class_{}'.format(i) for i in range(1, 11)],
                    index_label='Sample_id')
