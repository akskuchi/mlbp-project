import pandas as pd

import os


def write(pred_accuracy, pred_logloss):

    pred_accuracy = pd.DataFrame(pred_accuracy)
    pred_logloss = pd.DataFrame(pred_logloss)

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
                            header=['Class_{}'.format(i)
                                    for i in range(1, 11)],
                            index_label='Sample_id')
