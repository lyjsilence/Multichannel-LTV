import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import sklearn.metrics as metrics
from typing import Sequence


CATEGORICAL_FEATURES = ['chain', 'dept', 'category', 'brand', 'productmeasure']
NUMERIC_FEATURES = ['log_calibration_value']


def preprocess_single(data):
    feat_sizes = {}
    feat_sizes['cat_feat_sizes'] = []

    for key in CATEGORICAL_FEATURES:
        encoder = LabelEncoder()
        data[key] = encoder.fit_transform(data[key])
        feat_sizes['cat_feat_sizes'].append(data[key].nunique())

    feat_sizes['num_feat_sizes'] = len(NUMERIC_FEATURES)

    return data, feat_sizes


def preprocess_cross(data_dict):
    feat_sizes = {}
    feat_sizes['cat_feat_sizes'] = []

    data_list, encoder_list = [], []
    for id, data in data_dict.items():
        data_list.append(data)
    data_list = pd.concat(data_list, axis=0)

    for key in CATEGORICAL_FEATURES:
        feat_sizes['cat_feat_sizes'].append(data_list[key].nunique())
        encoder = LabelEncoder()
        data_list[key] = encoder.fit(data_list[key])
        encoder_list.append(encoder)


    for id, data in data_dict.items():
        for i, key in enumerate(CATEGORICAL_FEATURES):
            data[key] = encoder_list[i].transform(data[key])

    feat_sizes['num_feat_sizes'] = len(NUMERIC_FEATURES) * len(data_dict)

    return data_dict, feat_sizes


def train_val_test_split(data_dict, folds=5):
    user_id = data_dict[10000]['id'].tolist()
    # split train/test dataset
    train_val_ids, test_ids = train_test_split(user_id, test_size=0.2)

    # construct cross-validation
    kfold = KFold(n_splits=folds, shuffle=True, random_state=42)
    train_val_ids_fold = []
    for train_idx, test_idx in kfold.split(train_val_ids):
        train_ids = [train_val_ids[idx] for idx in train_idx]
        val_ids = [train_val_ids[idx] for idx in test_idx]
        train_val_ids_fold.append([train_ids, val_ids])

    return train_val_ids_fold, test_ids


class single_ltv_dataset(Dataset):
    def __init__(self, args, data, ids):
        self.data = data[data['id'].isin(ids)]
        map_dict = dict(zip(self.data["id"].unique(), np.arange(self.data["id"].nunique())))
        self.data["id"] = self.data["id"].map(map_dict)

        # the number of unique id
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, id):
        return self.data[self.data['id'] == id]


def single_ltv_collate_fn(batch):
    batch = pd.concat(batch, axis=0)

    x_cat = torch.IntTensor(batch[CATEGORICAL_FEATURES].values)
    x_num = torch.FloatTensor(batch[NUMERIC_FEATURES].values)
    y = torch.FloatTensor(batch['label'].values)

    res = {}
    res["x_cat"] = x_cat
    res["x_num"] = x_num
    res["y"] = y

    return res


class cross_ltv_dataset(Dataset):
    def __init__(self, args, data_dict, ids):
        flag = True
        for id, data in data_dict.items():
            if flag == True:
                data = data.reset_index(drop=True)
                data.columns = [col + f'_{str(id)}' if col != 'id' else col for col in data.columns]
                data_all = data
                flag = False
            else:
                data = data.reset_index(drop=True)
                data.columns = [col + f'_{str(id)}' if col != 'id' else col for col in data.columns]
                data_all = pd.merge(left=data_all, right=data, on='id', how='inner')

        self.data = data_all[data_all['id'].isin(ids)]
        map_dict = dict(zip(self.data["id"].unique(), np.arange(self.data["id"].nunique())))
        self.data['id'] = self.data['id'].map(map_dict)

        # the number of unique id
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, id):
        return self.data[self.data['id'] == id]


def cross_ltv_collate_fn(batch):
    batch = pd.concat(batch, axis=0)
    x_cat = []

    for cat_feat in CATEGORICAL_FEATURES:
        x_cat.append(torch.IntTensor(batch.loc[:, [c.startswith(cat_feat) for c in batch.columns]].values))

    x_num = torch.FloatTensor(batch.loc[:, [c.startswith(NUMERIC_FEATURES[0]) for c in batch.columns]].values)

    y = torch.FloatTensor(batch.loc[:, [c.startswith('label') for c in batch.columns]].values)

    res = {}
    res["x_cat"] = x_cat
    res["x_num"] = x_num
    res["y"] = y

    return res


def compute_metrics(x, y, model_name):
    if model_name == 'single_ltv':
        # get the predict parameters
        prob, loc, scale = x[:, 0], x[:, 1], x[:, 2]
        positive = torch.tensor(y > 0, dtype=torch.float32)

        prob = torch.sigmoid(prob).detach().cpu().numpy()
        positive = positive.detach().cpu().numpy()
        aucroc = [metrics.roc_auc_score(positive, prob)]
    else:
        aucroc = []
        for i in range(len(x)):
            x_s, y_s = x[i], y[:, i]
            # get the predict parameters
            prob, loc, scale = x_s[:, 0], x_s[:, 1], x_s[:, 2]
            positive = torch.tensor(y_s > 0, dtype=torch.float32)

            prob = torch.sigmoid(prob).detach().cpu().numpy()
            positive = positive.detach().cpu().numpy()
            aucroc.append(metrics.roc_auc_score(positive, prob))

    return aucroc


def cumulative_true(
    y_true: Sequence[float],
    y_pred: Sequence[float]
) -> np.ndarray:
    """Calculates cumulative sum of lifetime values over predicted rank.

    Arguments:
    y_true: true lifetime values.
    y_pred: predicted lifetime values.

    Returns:
    res: cumulative sum of lifetime values over predicted rank.
    """
    df = pd.DataFrame({
      'y_true': y_true,
      'y_pred': y_pred,
    }).sort_values(
      by='y_pred', ascending=False)

    return (df['y_true'].cumsum() / df['y_true'].sum()).values


def gini_from_gain(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates gini coefficient over gain charts.

    Arguments:
    df: Each column contains one gain chart. First column must be ground truth.

    Returns:
    gini_result: This dataframe has two columns containing raw and normalized
                 gini coefficient.
    """
    raw = df.apply(lambda x: 2 * x.sum() / df.shape[0] - 1.)
    normalized = raw / raw[0]
    return pd.DataFrame({
      'raw': raw,
      'normalized': normalized
    })[['raw', 'normalized']]


def compute_gini(y_pred, y, model_name):
    if model_name == 'single_ltv':
        gain = pd.DataFrame({
            'lorenz': cumulative_true(y.cpu().detach().numpy(), y.cpu().detach().numpy()),
            'model': cumulative_true(y.cpu().detach().numpy(), y_pred.cpu().detach().numpy()),
        })
        gini = [gini_from_gain(gain[['lorenz', 'model']])['normalized'][1]]
    else:
        gini = []
        for i in range(len(y_pred)):
            y_s, y_pred_s = y[:, i], y_pred[i]
            gain = pd.DataFrame({
                'lorenz': cumulative_true(y_s.cpu().detach().numpy(), y_s.cpu().detach().numpy()),
                'model': cumulative_true(y_s.cpu().detach().numpy(), y_pred_s.cpu().detach().numpy()),
            })
            gini.append(gini_from_gain(gain[['lorenz', 'model']])['normalized'][1])

    return gini
