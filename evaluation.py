import pandas as pd
import numpy as np


def build_pred_df(test_df, preds_list, strict=True):
    dataset_test_group = test_df.groupby('sentence_id', group_keys=False, as_index=False)[
        'words', 'labels'].agg(lambda x: list(x))
    y_test = dataset_test_group['labels']
    x_test = dataset_test_group['words']

    x_test_list = []
    y_pred_list = []
    y_test_list = []
    for row in range(0, len(preds_list)):
        x_test_list = np.concatenate((x_test_list, x_test[row]), axis=0)
        y_pred_list = np.concatenate((y_pred_list, preds_list[row]), axis=0)
        y_test_list = np.concatenate((y_test_list, y_test[row]), axis=0)
    data_prediction = pd.DataFrame(
        {'words': x_test_list, 'actual_class': y_test_list, 'predicted_class': y_pred_list})
    confusion_matrix = pd.crosstab(
        data_prediction['predicted_class'], data_prediction['actual_class'])
    if strict == True:
        data_prediction['strict'] = data_prediction.apply(
            lambda row: True if row.actual_class == row.predicted_class else False, axis=1)
    else:
        data_prediction['actual_non'] = data_prediction.apply(
            lambda row: row.actual_class.split('-')[-1], axis=1)
        data_prediction['predicted_non'] = data_prediction.apply(
            lambda row: row.predicted_class.split('-')[-1], axis=1)
        data_prediction['non_strict'] = data_prediction.apply(lambda row: True if row.actual_class.split(
            '-')[-1] == row.predicted_class.split('-')[-1] else False, axis=1)
        confusion_matrix = pd.crosstab(
            data_prediction['predicted_non'], data_prediction['actual_non'])
    return data_prediction, confusion_matrix


def get_confusion_matrix(confusion_matrix):
    sum_test = confusion_matrix.sum(axis=0, numeric_only=True).to_list()
    sum_pred = confusion_matrix.sum(axis=1, numeric_only=True).to_list()
    tp_index = [index for index in range(0, confusion_matrix.shape[0])]
    tp = [confusion_matrix.iloc[index, index + 1] for index in tp_index]
    confusion_matrix['TRUE'] = sum_test
    confusion_matrix['PRED'] = sum_pred
    confusion_matrix['TP'] = tp
    fp = list(np.subtract(np.array(sum_pred), np.array(tp)))
    fn = list(np.subtract(np.array(sum_test), np.array(tp)))
    confusion_matrix['FP'] = fp
    confusion_matrix['FN'] = fn
    sum_tp = confusion_matrix['TP'].sum()
    tn = [(sum_tp - tp[index]) for index in range(0, len(tp))]
    confusion_matrix['TN'] = tn
    sum_vertical1 = confusion_matrix.sum(axis=0, numeric_only=True).to_list()
    sum_vertical1.insert(0, 'SUM')

    confusion_matrix.loc[len(confusion_matrix.index)] = sum_vertical1

    return confusion_matrix


def compute_score(tp, fp, fn, tn):
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)
    return accuracy, precision, recall, f1


def get_tp_fp_fn_tn(confusion_matrix, entity='O'):
    tp = confusion_matrix[confusion_matrix['entity'] == entity]['TP'].sum()
    fp = confusion_matrix[confusion_matrix['entity'] == entity]['FP'].sum()
    fn = confusion_matrix[confusion_matrix['entity'] == entity]['FN'].sum()
    tn = confusion_matrix[confusion_matrix['entity'] == entity]['TN'].sum()
    return tp, fp, fn, tn


def get_evaluation_score(confusion_matrix):
    eval = dict()
    confusion_matrix['entity'] = list(entity.split('-')[-1]
                                      for entity in confusion_matrix.iloc[:, 0].unique())
    entity_types = list(set([entity.split('-')[-1]
                        for entity in confusion_matrix.iloc[:, 0].unique() if entity != 'SUM']))
    for entity in entity_types:
        tp, fp, fn, tn = get_tp_fp_fn_tn(confusion_matrix, entity)
        accuracy, precision, recall, f1 = compute_score(tp, fp, fn, tn)
        eval[entity] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    per_class = pd.DataFrame(eval)
    per_class = per_class.transpose().mean()
    eval['macro'] = {
        'accuracy': per_class[0],
        'precision': per_class[1],
        'recall': per_class[2],
        'f1': per_class[3],
    }
    tp, fp, fn, tn = get_tp_fp_fn_tn(confusion_matrix, 'SUM')
    accuracy, precision, recall, f1 = compute_score(tp, fp, fn, tn)
    eval['micro'] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    overall_score = pd.DataFrame(eval).transpose()
    return eval, overall_score

