from sklearn import tree, linear_model
import os
import logging
import numpy
import pandas

# create logger
logger = logging.getLogger('simple_predictor')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

train_data_features = pandas.DataFrame()
train_data_labels = pandas.DataFrame()
test_data_features = pandas.DataFrame()


def read_train_data_features(file_location):
    """
    :param file_location: features for each song (describing timbre, pitch, rhythm)
    """
    logger.info('reading features of songs from the training data')
    try:
        global train_data_features
        train_data_features = pandas.read_csv(os.path.join(os.path.dirname(__file__), '..', file_location))
        logger.info('shape of the train data features set: (%s, %s)' % train_data_features.shape)
    except Exception as read_exception:
        logger.error(read_exception)


def read_train_data_labels(file_location):
    """
    :param file_location: labels corresponding to each song (denoting the genre of the song)
    """
    logger.info('reading corresponding labels of songs from the training data')
    try:
        global train_data_labels
        train_data_labels = pandas.read_csv(os.path.join(os.path.dirname(__file__), '..', file_location))
        logger.info('shape of the train data labels set: %d with %d distinct song genres' %
                    (train_data_labels.shape[0], numpy.unique(train_data_labels).shape[0]))
    except Exception as read_exception:
        logger.error(read_exception)


def read_test_data_features(file_location):
    """
    :param file_location: features of the test data set
    """
    logger.info('reading features of songs for the test data')
    try:
        global test_data_features
        test_data_features = pandas.read_csv(os.path.join(os.path.dirname(__file__), '..', file_location))
        logger.info('shape of the test data features set: (%s, %s)' % test_data_features.shape)
    except Exception as read_exception:
        logger.error(read_exception)


# TODO: weights modification based on observations
def predict_genres_decision_tree():
    """
    based on the standard ski-kit-learn decsision trees:
    http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    """

    global test_data_features, train_data_features, train_data_labels

    classifier = tree.DecisionTreeClassifier()
    try:
        logger.info('training decision tree classifier of sklearn')
        classifier.fit(train_data_features, train_data_labels.values.ravel())
        logger.info('training complete, let\'s predict')
    except Exception as training_error:
        logger.error(training_error)

    try:
        prediction = classifier.predict(test_data_features.head(5))
        pandas.DataFrame({'Sample_id': range(1, len(prediction) + 1), 'Sample_label': prediction}).set_index('Sample_id').to_csv(
            '../accuracy_output.csv',
            sep=' ')
        logger.info('prediction complete, results exported to file: accuracy.csv')
    except Exception as prediction_error:
        logger.error(prediction_error)


# TODO: weights modification based on observations
def predict_genres_logistic_regression():
    """
    based on the standard ski-kit-learn logistic regression:
    http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """

    global test_data_features, train_data_features, train_data_labels

    logistic_regressor = linear_model.LogisticRegression()
    try:
        logger.info('training the logistic regressor of sklearn')
        logistic_regressor.fit(train_data_features, train_data_labels.values.ravel())
        logger.info('training complete, let\'s predict')
    except Exception as training_error:
        logger.error(training_error)

    try:
        prediction = logistic_regressor.predict(test_data_features.head(1))
        pandas.DataFrame({'Sample_id': range(1, len(prediction) + 1), 'Sample_label': prediction}).set_index('Sample_id').to_csv(
            '../accuracy_output.csv',
            sep=' ')
        logger.info('prediction complete, results exported to file: accuracy.csv')
    except Exception as prediction_error:
        logger.error(prediction_error)


if __name__ == '__main__':
    # data
    read_train_data_features('resources/train_data.csv')
    read_train_data_labels('resources/train_labels.csv')
    read_test_data_features('resources/test_data.csv')

    # model and predict
    predict_genres_decision_tree()
    predict_genres_logistic_regression()

    # loss/risk analysis
