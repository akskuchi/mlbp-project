import numpy as np
from sklearn import preprocessing, linear_model, pipeline,\
    model_selection

import load_data
import write_data

write_output = False
accuracy = True
logloss = True

X, y, test_data = load_data.load()

# common pipeline for accuracy and log-loss
pl = pipeline.Pipeline([
    ('scaler', preprocessing.StandardScaler()),
    ('classifier', None)
])

# common k-fold object
kf = model_selection.StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=1
)

############### ACCURACY ###############

if accuracy:

    gs_acc = model_selection.GridSearchCV(
        estimator=pl,
        param_grid=dict(
            classifier=[
                linear_model.LogisticRegression(
                    random_state=1
                )
            ],
            classifier__C=[1]  # [0.1, 0.2, 0.5, 0.8, 1, 1.2, 1.5, 2, 5]
        ),
        scoring='accuracy',
        cv=kf)

    # Use 3x3 cross-validation to estimate accuracy
    scores_accuracy = model_selection.cross_val_score(
        estimator=gs_acc, X=X, y=y, scoring='accuracy',
        cv=kf
    )

    # use all available data to fit a model
    gs_acc.fit(X, y)

    print('\n')
    print('---------- ACCURACY ----------')
    print('CV scores: {}'.format(scores_accuracy))
    print('Average: {0:.3f} +/- {1:.4f}'
          .format(np.mean(scores_accuracy), np.std(scores_accuracy)))
    print(gs_acc.best_estimator_)
    print('------------------------------')

############### LOGLOSS ###############

if logloss:

    gs_log = model_selection.GridSearchCV(
        estimator=pl,
        param_grid=dict(
            classifier=[
                linear_model.LogisticRegression(
                    random_state=1,
                    solver='newton-cg',
                    multi_class='multinomial'
                )
            ],
            classifier__C=[1]
        ),
        scoring='neg_log_loss',
        cv=kf
    )

    # Use 3x3 cross-validation to estimate neg. log-loss
    scores_logloss = model_selection.cross_val_score(
        gs_log, X, y,
        scoring='neg_log_loss',
        cv=kf
    )

    # use all available data to fit a model
    gs_log.fit(X, y)

    print('---------- LOG_LOSS ----------')
    print('CV scores: {}'.format(scores_logloss))
    print('Average: {0:.3f} +/- {1:.4f}'
          .format(np.mean(scores_logloss), np.std(scores_logloss)))
    print(gs_log.best_estimator_)
    print('------------------------------')


if write_output and accuracy and logloss:
    pred_accuracy = gs_acc.predict(test_data)
    pred_logloss = gs_log.predict_proba(test_data)
    write_data.write(pred_accuracy, pred_logloss)
