"""
Payment-level auxiliary advertiser experiment for the Multichannel-LTV project.

Goal
----
For a fixed target company, test whether sharing data with advertisers from
high-, middle-, or low-payment groups changes target-company prediction accuracy.

Protocol
--------
1. Compute each company's average user payment level:
       mean_user_payment = mean over users of sum(label) within that company.
   If the data has one row per user/company, this is equivalent to mean(label).

2. Rank the 20 companies by mean_user_payment and split them into three groups:
       low_payment, mid_payment, high_payment.

3. For a given target_company_id, train three cross-advertiser models:
       target company + 2 auxiliary companies from low_payment
       target company + 2 auxiliary companies from mid_payment
       target company + 2 auxiliary companies from high_payment

4. Repeat the auxiliary sampling multiple times. Within each payment group,
   auxiliary pairs are sampled reproducibly and, when possible, without repeated
   pairs before all possible pairs have been used.

5. The target company's train/validation/test users are fixed across all payment
   groups and repeats. This isolates the effect of auxiliary advertiser type.

Example
-------
python run_payment_level_auxiliary_experiment_gpu.py \
    --data_path data.pickle \
    --target_company_id 10000 \
    --num_aux 2 \
    --sample_repeats 10 \
    --fold 5 \
    --epoch_max 50 \
    --device auto \
    --num_workers 2 \
    --seed 0

Main outputs
------------
results_payment_level_auxiliary/
    company_payment_groups.csv
    auxiliary_samples.csv
    results_target_long.csv
    results_target_aggregate.csv
    results_all_tasks_long.csv
"""

import argparse
import hashlib
import itertools
import os
import pickle
import random
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset


CATEGORICAL_FEATURES = ['chain', 'dept', 'category', 'brand', 'productmeasure']
NUMERIC_FEATURES = ['log_calibration_value']
LABEL_COL = 'label'
ID_COL = 'id'
PAYMENT_GROUP_ORDER = ['low_payment', 'mid_payment', 'high_payment']


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

SEED_MODULUS = 2 ** 32 - 1


def normalize_seed(seed: int) -> int:
    """Map any integer seed into NumPy's valid seed range."""
    return int(seed) % SEED_MODULUS


def make_reproducible_seed(*components: object) -> int:
    """Create a stable 32-bit seed from arbitrary components.

    This avoids NumPy's "Seed must be between 0 and 2**32 - 1" error
    when company ids are large. It is deterministic across Python sessions,
    unlike Python's built-in hash().
    """
    text = '||'.join(str(x) for x in components)
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    return int(digest[:16], 16) % SEED_MODULUS


def set_seed(seed: int, strict_deterministic: bool = False) -> None:
    """Set Python/NumPy/PyTorch seeds with a NumPy-safe seed value."""
    safe_seed = normalize_seed(seed)
    random.seed(safe_seed)
    np.random.seed(safe_seed)
    torch.manual_seed(safe_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(safe_seed)
        torch.cuda.manual_seed_all(safe_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if strict_deterministic:
        torch.use_deterministic_algorithms(True)


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def move_batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    non_blocking: bool = True,
) -> Dict[str, torch.Tensor]:
    """Move a collated batch to CPU/GPU once before model execution.

    The original repository moved individual tensors inside the model. For CUDA,
    moving the whole batch with non_blocking=True and pin_memory=True in the
    DataLoader usually reduces CPU-to-GPU transfer overhead.
    """
    if device.type == 'cuda':
        return {
            'x_cat': [x.to(device, non_blocking=non_blocking) for x in batch['x_cat']],
            'x_num': batch['x_num'].to(device, non_blocking=non_blocking),
            'y': batch['y'].to(device, non_blocking=non_blocking),
        }
    return {
        'x_cat': [x.to(device) for x in batch['x_cat']],
        'x_num': batch['x_num'].to(device),
        'y': batch['y'].to(device),
    }


def parse_group_list(value: str) -> List[str]:
    if value.lower() == 'all':
        return PAYMENT_GROUP_ORDER.copy()
    groups = [x.strip() for x in value.split(',') if x.strip()]
    unknown = [g for g in groups if g not in PAYMENT_GROUP_ORDER]
    if unknown:
        raise ValueError('Unknown payment groups: {}. Valid groups are {}'.format(unknown, PAYMENT_GROUP_ORDER))
    return groups


def get_common_user_ids(data_dict: Dict[int, pd.DataFrame], company_ids: Sequence[int]) -> List[int]:
    """Return user ids appearing in all selected companies."""
    common_ids = None
    for cid in company_ids:
        ids = set(data_dict[cid][ID_COL].tolist())
        if common_ids is None:
            common_ids = ids
        else:
            common_ids = common_ids.intersection(ids)
    return sorted(list(common_ids))


def make_fixed_splits(
    data_dict: Dict[int, pd.DataFrame],
    all_company_ids: Sequence[int],
    folds: int,
    seed: int,
    test_size: float,
) -> Tuple[List[Tuple[List[int], List[int]]], List[int]]:
    """Create fixed train/validation/test user splits shared by all conditions.

    The split is based on users common to all companies. Therefore every model
    condition is evaluated on exactly the same target-company test users.
    """
    user_ids = get_common_user_ids(data_dict, all_company_ids)
    if len(user_ids) == 0:
        raise ValueError('No common user ids across companies. Cannot create fixed cross-company split.')

    train_val_ids, test_ids = train_test_split(
        user_ids, test_size=test_size, random_state=seed, shuffle=True
    )

    kfold = KFold(n_splits=folds, shuffle=True, random_state=seed)
    train_val_ids = np.array(train_val_ids)
    train_val_ids_fold = []
    for train_idx, val_idx in kfold.split(train_val_ids):
        train_ids = train_val_ids[train_idx].tolist()
        val_ids = train_val_ids[val_idx].tolist()
        train_val_ids_fold.append((train_ids, val_ids))

    return train_val_ids_fold, list(test_ids)


# -----------------------------------------------------------------------------
# Payment-level grouping and auxiliary sampling
# -----------------------------------------------------------------------------

def compute_company_payment_summary(data_dict: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Compute company-level average user payment.

    This first aggregates label at the user level and then averages across users.
    It is robust to either one row per user/company or multiple rows per user.
    """
    rows = []
    for cid, df in data_dict.items():
        if ID_COL not in df.columns or LABEL_COL not in df.columns:
            raise ValueError('Company {} data must contain columns {} and {}'.format(cid, ID_COL, LABEL_COL))
        user_payment = df.groupby(ID_COL)[LABEL_COL].sum()
        rows.append({
            'company_id': int(cid),
            'num_rows': int(len(df)),
            'num_users': int(user_payment.shape[0]),
            'mean_user_payment': float(user_payment.mean()),
            'median_user_payment': float(user_payment.median()),
            'positive_user_rate': float((user_payment > 0).mean()),
        })
    summary = pd.DataFrame(rows).sort_values(['mean_user_payment', 'company_id']).reset_index(drop=True)
    return summary


def assign_payment_groups(payment_summary: pd.DataFrame) -> pd.DataFrame:
    """Split companies into low/mid/high groups by ranked average payment.

    With 20 companies, the group sizes will be as balanced as possible. For
    example, np.array_split gives 7 low, 7 middle, and 6 high companies after
    sorting by mean_user_payment in ascending order.
    """
    summary = payment_summary.sort_values(['mean_user_payment', 'company_id']).reset_index(drop=True).copy()
    splits = np.array_split(summary.index.to_numpy(), 3)

    summary['payment_group'] = None
    summary['payment_rank_ascending'] = np.arange(1, len(summary) + 1)
    for group_name, idx in zip(PAYMENT_GROUP_ORDER, splits):
        summary.loc[idx, 'payment_group'] = group_name

    return summary


def make_auxiliary_sets_for_group(
    group_company_ids: Sequence[int],
    target_company_id: int,
    num_aux: int,
    repeats: int,
    seed: int,
    group_name: str,
) -> List[List[int]]:
    """Randomly sample auxiliary company sets from one payment group.

    Sampling is reproducible. It avoids repeated auxiliary combinations until
    all combinations from that group have been used. If repeats exceeds the
    number of possible combinations, the shuffled combination list is recycled.
    """
    if repeats < 1:
        raise ValueError('sample_repeats must be >= 1')
    if num_aux < 1:
        raise ValueError('num_aux must be >= 1')

    candidates = [int(cid) for cid in group_company_ids if int(cid) != int(target_company_id)]
    candidates = sorted(candidates)

    if len(candidates) < num_aux:
        raise ValueError(
            'Payment group {} has only {} eligible auxiliary companies after excluding target {}, '
            'but num_aux={} was requested.'.format(group_name, len(candidates), target_company_id, num_aux)
        )

    combinations = [list(x) for x in itertools.combinations(candidates, num_aux)]
    rng_seed = make_reproducible_seed('auxiliary_sets', seed, target_company_id, group_name, num_aux)
    rng = np.random.RandomState(rng_seed)

    sampled = []
    while len(sampled) < repeats:
        order = np.arange(len(combinations))
        rng.shuffle(order)
        for idx in order:
            sampled.append(combinations[int(idx)])
            if len(sampled) >= repeats:
                break

    return sampled[:repeats]


# -----------------------------------------------------------------------------
# Encoding
# -----------------------------------------------------------------------------

def fit_global_encoders(
    data_dict: Dict[int, pd.DataFrame],
    company_ids: Sequence[int],
) -> Tuple[Dict[str, LabelEncoder], Dict[str, List[int]]]:
    """Fit categorical encoders once on all companies."""
    encoders = {}
    feat_sizes = {'cat_feat_sizes': []}

    for feature in CATEGORICAL_FEATURES:
        values = []
        for cid in company_ids:
            values.append(data_dict[cid][feature].astype(str))
        values = pd.concat(values, axis=0)

        encoder = LabelEncoder()
        encoder.fit(values)
        encoders[feature] = encoder
        feat_sizes['cat_feat_sizes'].append(len(encoder.classes_))

    return encoders, feat_sizes


def encode_selected_companies(
    data_dict: Dict[int, pd.DataFrame],
    selected_company_ids: Sequence[int],
    encoders: Dict[str, LabelEncoder],
) -> Tuple[Dict[int, pd.DataFrame], Dict[str, List[int]]]:
    """Return encoded copies for selected companies only."""
    encoded = {}

    feat_sizes = {
        'cat_feat_sizes': [len(encoders[feature].classes_) for feature in CATEGORICAL_FEATURES],
        'num_feat_sizes': len(NUMERIC_FEATURES) * len(selected_company_ids),
    }

    needed_cols = [ID_COL] + CATEGORICAL_FEATURES + NUMERIC_FEATURES + [LABEL_COL]
    for cid in selected_company_ids:
        df = data_dict[cid][needed_cols].copy()
        for feature in CATEGORICAL_FEATURES:
            df[feature] = encoders[feature].transform(df[feature].astype(str))
        encoded[int(cid)] = df

    return encoded, feat_sizes


# -----------------------------------------------------------------------------
# Dataset and dynamic cross-advertiser model
# -----------------------------------------------------------------------------

class CrossAdvertiserDataset(Dataset):
    """Merged user-level dataset for an arbitrary subset of companies."""

    def __init__(
        self,
        data_dict: Dict[int, pd.DataFrame],
        company_ids: Sequence[int],
        user_ids: Sequence[int],
    ) -> None:
        self.company_ids = [int(cid) for cid in company_ids]
        user_id_set = set(user_ids)

        merged = None
        for cid in self.company_ids:
            df = data_dict[cid].copy()
            df = df[df[ID_COL].isin(user_id_set)].copy()
            rename = {col: f'{col}_{cid}' for col in df.columns if col != ID_COL}
            df = df.rename(columns=rename)
            if merged is None:
                merged = df
            else:
                merged = pd.merge(merged, df, on=ID_COL, how='inner')

        if merged is None or len(merged) == 0:
            raise ValueError('No rows remain after merging selected companies. Check user ids and company ids.')

        self.data = merged.sort_values(ID_COL).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> pd.DataFrame:
        return self.data.iloc[[idx]]


def make_cross_collate_fn(company_ids: Sequence[int]):
    company_ids = [int(cid) for cid in company_ids]

    def collate_fn(batch: List[pd.DataFrame]) -> Dict[str, torch.Tensor]:
        batch_df = pd.concat(batch, axis=0)

        x_cat = []
        for cat_feature in CATEGORICAL_FEATURES:
            cols = [f'{cat_feature}_{cid}' for cid in company_ids]
            x_cat.append(torch.LongTensor(batch_df[cols].values))

        numeric_cols = []
        for cid in company_ids:
            for num_feature in NUMERIC_FEATURES:
                numeric_cols.append(f'{num_feature}_{cid}')
        x_num = torch.FloatTensor(batch_df[numeric_cols].values)

        y_cols = [f'{LABEL_COL}_{cid}' for cid in company_ids]
        y = torch.FloatTensor(batch_df[y_cols].values)

        return {'x_cat': x_cat, 'x_num': x_num, 'y': y}

    return collate_fn


class CrossLTVDynamic(nn.Module):
    """Original cross_ltv model generalized from exactly 20 tasks to N tasks."""

    def __init__(self, args: argparse.Namespace, feat_sizes: Dict[str, List[int]], device: torch.device):
        super(CrossLTVDynamic, self).__init__()
        categorical_feature_sizes = feat_sizes['cat_feat_sizes']
        numeric_feature_size = feat_sizes['num_feat_sizes']
        numeric_dim = 10 * numeric_feature_size

        self.num_tasks = args.num_tasks
        self.num_experts = args.num_experts
        self.device = device

        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=size, embedding_dim=max(1, int(np.log2(size))))
            for size in categorical_feature_sizes
        ])

        self.numeric = nn.Linear(numeric_feature_size, numeric_dim)

        total_emb_size = sum(e.embedding_dim * self.num_tasks for e in self.embeddings)
        input_size = total_emb_size + numeric_dim
        expert_units = max(1, input_size // 2)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, expert_units),
                nn.ReLU(),
                nn.Dropout(args.dropout),
            )
            for _ in range(self.num_experts)
        ])

        self.gates = nn.ModuleList([
            nn.Linear(input_size, args.num_experts) for _ in range(self.num_tasks)
        ])
        self.task_layers = nn.ModuleList([
            nn.Linear(expert_units, 3) for _ in range(self.num_tasks)
        ])

    def forward(self, x_cat_list: List[torch.Tensor], x_num: torch.Tensor) -> List[torch.Tensor]:
        embeddings_list = []
        for i, x_cat in enumerate(x_cat_list):
            if x_cat.device != self.device:
                x_cat = x_cat.to(self.device)
            embeddings = [self.embeddings[i](x_cat[:, task_idx]) for task_idx in range(x_cat.shape[1])]
            embeddings_list.append(torch.cat(embeddings, dim=1))

        x_cat = torch.cat(embeddings_list, dim=1)
        if x_num.device != self.device:
            x_num = x_num.to(self.device)
        x_num = self.numeric(x_num)
        x = torch.cat([x_cat, x_num], dim=1)

        expert_outputs = [expert(x) for expert in self.experts]

        task_outputs = []
        for task_idx in range(self.num_tasks):
            gate_outputs = F.softmax(self.gates[task_idx](x), dim=1)
            weighted_expert_output = sum(
                gate_outputs[:, expert_idx:expert_idx + 1] * expert_outputs[expert_idx]
                for expert_idx in range(self.num_experts)
            )
            task_outputs.append(self.task_layers[task_idx](weighted_expert_output))

        return task_outputs

    def compute_loss(self, x_all: List[torch.Tensor], y_all: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        criterion = nn.BCEWithLogitsLoss()
        classification_loss = 0.0
        regression_loss = 0.0
        if y_all.device != self.device:
            y_all = y_all.to(self.device)

        for task_idx in range(self.num_tasks):
            x = x_all[task_idx]
            y = y_all[:, task_idx]

            prob, loc, scale = x[:, 0], x[:, 1], x[:, 2]
            scale = F.softplus(scale)

            positive = (y > 0).float()
            safe_labels = positive * y + (1.0 - positive) * torch.ones_like(y)

            classification_loss = classification_loss + criterion(prob, positive)
            regression_loss = regression_loss - torch.mean(
                positive * torch.distributions.LogNormal(loc=loc, scale=scale).log_prob(safe_labels)
            )

        return classification_loss / self.num_tasks, regression_loss / self.num_tasks

    def predict(self, x_all: List[torch.Tensor]) -> List[torch.Tensor]:
        pred_list = []
        for task_idx in range(self.num_tasks):
            x = x_all[task_idx]
            prob, loc, scale = x[:, 0], x[:, 1], x[:, 2]
            scale = F.softplus(scale)
            # Keep the original repository's prediction rule for comparability.
            preds = prob * torch.exp(loc + 0.5 * torch.pow(scale, 2))
            pred_list.append(preds)
        return pred_list


# -----------------------------------------------------------------------------
# Metrics and training
# -----------------------------------------------------------------------------

def cumulative_true(y_true: Sequence[float], y_pred: Sequence[float]) -> np.ndarray:
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}).sort_values(by='y_pred', ascending=False)
    total = df['y_true'].sum()
    if total == 0:
        return np.zeros(len(df), dtype=float)
    return (df['y_true'].cumsum() / total).values


def gini_from_gain(df: pd.DataFrame) -> pd.DataFrame:
    raw = df.apply(lambda x: 2 * x.sum() / df.shape[0] - 1.0)
    if raw.iloc[0] == 0:
        normalized = raw * np.nan
    else:
        normalized = raw / raw.iloc[0]
    return pd.DataFrame({'raw': raw, 'normalized': normalized})[['raw', 'normalized']]


def compute_auc_and_gini(y_true: np.ndarray, prob_logits: np.ndarray, ltv_pred: np.ndarray) -> Tuple[float, float]:
    positive = (y_true > 0).astype(float)
    prob = 1.0 / (1.0 + np.exp(-prob_logits))

    if len(np.unique(positive)) < 2:
        auc = np.nan
    else:
        auc = metrics.roc_auc_score(positive, prob)

    gain = pd.DataFrame({
        'lorenz': cumulative_true(y_true, y_true),
        'model': cumulative_true(y_true, ltv_pred),
    })
    gini = gini_from_gain(gain[['lorenz', 'model']])['normalized'].iloc[1]
    return float(auc), float(gini)


def evaluate_model(
    model: CrossLTVDynamic,
    dl: DataLoader,
    device: torch.device,
    company_ids: Sequence[int],
    target_company_id: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    model.eval()
    all_logits = [[] for _ in company_ids]
    all_preds = [[] for _ in company_ids]
    all_y = [[] for _ in company_ids]
    losses = []

    with torch.no_grad():
        for batch in dl:
            batch = move_batch_to_device(batch, device, non_blocking=True)
            x_all = model(batch['x_cat'], batch['x_num'])
            class_loss, reg_loss = model.compute_loss(x_all, batch['y'])
            losses.append((class_loss + reg_loss).item())
            preds = model.predict(x_all)

            y = batch['y'].cpu().numpy()
            for task_idx, _ in enumerate(company_ids):
                all_logits[task_idx].append(x_all[task_idx][:, 0].detach().cpu().numpy())
                all_preds[task_idx].append(preds[task_idx].detach().cpu().numpy())
                all_y[task_idx].append(y[:, task_idx])

    rows = []
    for task_idx, cid in enumerate(company_ids):
        y_true = np.concatenate(all_y[task_idx], axis=0)
        prob_logits = np.concatenate(all_logits[task_idx], axis=0)
        ltv_pred = np.concatenate(all_preds[task_idx], axis=0)
        auc, gini = compute_auc_and_gini(y_true, prob_logits, ltv_pred)
        rows.append({'company_id': int(cid), 'task_idx': task_idx, 'AUCROC': auc, 'GINI': gini})

    all_task_df = pd.DataFrame(rows)
    target_row = all_task_df[all_task_df['company_id'] == int(target_company_id)].iloc[0].to_dict()
    target_row['loss'] = float(np.mean(losses)) if len(losses) > 0 else np.nan
    return target_row, all_task_df


def train_one_run(
    args: argparse.Namespace,
    data_dict: Dict[int, pd.DataFrame],
    encoders: Dict[str, LabelEncoder],
    train_ids: Sequence[int],
    val_ids: Sequence[int],
    test_ids: Sequence[int],
    selected_company_ids: Sequence[int],
    target_company_id: int,
    run_save_path: str,
    device: torch.device,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    encoded_data, feat_sizes = encode_selected_companies(data_dict, selected_company_ids, encoders)

    collate_fn = make_cross_collate_fn(selected_company_ids)
    train_dataset = CrossAdvertiserDataset(encoded_data, selected_company_ids, train_ids)
    val_dataset = CrossAdvertiserDataset(encoded_data, selected_company_ids, val_ids)
    test_dataset = CrossAdvertiserDataset(encoded_data, selected_company_ids, test_ids)

    loader_common_kwargs = {
        'num_workers': args.num_workers,
        # On Windows, pin_memory can occasionally trigger low-level CUDA/DataLoader
        # instability. Keep it configurable instead of forcing it on.
        'pin_memory': bool(args.pin_memory and device.type == 'cuda'),
    }
    if args.num_workers > 0:
        loader_common_kwargs['persistent_workers'] = True
        loader_common_kwargs['prefetch_factor'] = args.prefetch_factor

    dl_train = DataLoader(
        dataset=train_dataset,
        collate_fn=collate_fn,
        shuffle=True,
        batch_size=args.batch_size,
        **loader_common_kwargs,
    )
    dl_val = DataLoader(
        dataset=val_dataset,
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=args.eval_batch_size,
        **loader_common_kwargs,
    )
    dl_test = DataLoader(
        dataset=test_dataset,
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=args.eval_batch_size,
        **loader_common_kwargs,
    )

    args.num_tasks = len(selected_company_ids)
    model = CrossLTVDynamic(args, feat_sizes, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.lr_patience, min_lr=args.min_lr
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == 'cuda'))

    ensure_dir(run_save_path)
    best_path = os.path.join(run_save_path, 'cross_ltv_payment_group_best.pkl')
    best_val_loss = np.inf
    log_rows = []

    for epoch in range(1, args.epoch_max + 1):
        model.train()
        epoch_start = time.time()
        train_class_losses = []
        train_reg_losses = []

        for batch_idx, batch in enumerate(dl_train, start=1):
            batch = move_batch_to_device(batch, device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if args.amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    x_all = model(batch['x_cat'], batch['x_num'])
                    class_loss, reg_loss = model.compute_loss(x_all, batch['y'])
                    loss = class_loss + reg_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                x_all = model(batch['x_cat'], batch['x_num'])
                class_loss, reg_loss = model.compute_loss(x_all, batch['y'])
                loss = class_loss + reg_loss
                loss.backward()
                optimizer.step()

            train_class_losses.append(class_loss.item())
            train_reg_losses.append(reg_loss.item())

            if args.verbose:
                print(
                    '\rEpoch [{}/{}], Batch [{}/{}], Class Loss: {:.4f}, Reg Loss: {:.4f}'.format(
                        epoch,
                        args.epoch_max,
                        batch_idx,
                        len(dl_train),
                        np.mean(train_class_losses),
                        np.mean(train_reg_losses),
                    ),
                    end='',
                )

        target_val, _ = evaluate_model(model, dl_val, device, selected_company_ids, target_company_id)
        val_loss = target_val['loss']
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

        row = {
            'epoch': epoch,
            'train_class_loss': float(np.mean(train_class_losses)),
            'train_reg_loss': float(np.mean(train_reg_losses)),
            'val_loss': float(val_loss),
            'val_target_AUCROC': target_val['AUCROC'],
            'val_target_GINI': target_val['GINI'],
            'time_elapsed': time.time() - epoch_start,
        }
        log_rows.append(row)

        if args.verbose:
            print(
                ' | Val target AUC: {:.4f}, GINI: {:.4f}, Loss: {:.4f}'.format(
                    target_val['AUCROC'], target_val['GINI'], val_loss
                )
            )

    model.load_state_dict(torch.load(best_path, map_location=device))
    target_test, all_task_test = evaluate_model(model, dl_test, device, selected_company_ids, target_company_id)
    log_df = pd.DataFrame(log_rows)

    log_df.to_csv(os.path.join(run_save_path, 'val_log.csv'), index=False)
    all_task_test.to_csv(os.path.join(run_save_path, 'test_results_all_tasks.csv'), index=False)

    return target_test, all_task_test, log_df



# -----------------------------------------------------------------------------
# Statistical testing: top group vs. second-best group
# -----------------------------------------------------------------------------

def parse_metric_list(value: str) -> List[str]:
    metrics = [x.strip() for x in value.split(',') if x.strip()]
    if not metrics:
        raise ValueError('At least one metric must be provided in --stat_test_metrics.')
    return metrics


def paired_sign_flip_permutation_test(
    x: Sequence[float],
    y: Sequence[float],
    n_permutations: int,
    seed: int,
) -> Dict[str, float]:
    """Paired sign-flip permutation test for mean(x - y)."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    diffs = x_arr[mask] - y_arr[mask]

    if diffs.size == 0:
        return {
            'n_paired': 0,
            'observed_mean_diff': np.nan,
            'permutation_p_one_sided': np.nan,
            'permutation_p_two_sided': np.nan,
        }

    observed = float(np.mean(diffs))
    if n_permutations <= 0:
        return {
            'n_paired': int(diffs.size),
            'observed_mean_diff': observed,
            'permutation_p_one_sided': np.nan,
            'permutation_p_two_sided': np.nan,
        }

    rng = np.random.RandomState(make_reproducible_seed('paired_permutation', seed, diffs.size, observed))
    null_means = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        signs = rng.choice(np.array([-1.0, 1.0]), size=diffs.size, replace=True)
        null_means[i] = np.mean(signs * diffs)

    p_one = (np.sum(null_means >= observed) + 1.0) / (n_permutations + 1.0)
    p_two = (np.sum(np.abs(null_means) >= abs(observed)) + 1.0) / (n_permutations + 1.0)

    return {
        'n_paired': int(diffs.size),
        'observed_mean_diff': observed,
        'permutation_p_one_sided': float(p_one),
        'permutation_p_two_sided': float(p_two),
    }


def scipy_paired_tests(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """Run paired t-test and Wilcoxon signed-rank test if SciPy is available."""
    out = {
        'paired_t_stat': np.nan,
        'paired_t_p_one_sided': np.nan,
        'paired_t_p_two_sided': np.nan,
        'wilcoxon_stat': np.nan,
        'wilcoxon_p_one_sided': np.nan,
        'wilcoxon_p_two_sided': np.nan,
    }

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 2:
        return out

    try:
        from scipy import stats
    except Exception:
        return out

    try:
        t_two = stats.ttest_rel(x_arr, y_arr, nan_policy='omit')
        out['paired_t_stat'] = float(t_two.statistic)
        out['paired_t_p_two_sided'] = float(t_two.pvalue)
        try:
            t_one = stats.ttest_rel(x_arr, y_arr, nan_policy='omit', alternative='greater')
            out['paired_t_p_one_sided'] = float(t_one.pvalue)
        except TypeError:
            if np.isfinite(t_two.statistic) and np.isfinite(t_two.pvalue):
                out['paired_t_p_one_sided'] = float(t_two.pvalue / 2.0) if t_two.statistic > 0 else float(1.0 - t_two.pvalue / 2.0)
    except Exception:
        pass

    try:
        diffs = x_arr - y_arr
        if np.any(np.abs(diffs) > 0):
            try:
                w_one = stats.wilcoxon(x_arr, y_arr, alternative='greater', zero_method='wilcox', method='auto')
            except TypeError:
                w_one = stats.wilcoxon(x_arr, y_arr, alternative='greater', zero_method='wilcox')
            out['wilcoxon_stat'] = float(w_one.statistic)
            out['wilcoxon_p_one_sided'] = float(w_one.pvalue)

            try:
                w_two = stats.wilcoxon(x_arr, y_arr, alternative='two-sided', zero_method='wilcox', method='auto')
            except TypeError:
                w_two = stats.wilcoxon(x_arr, y_arr, alternative='two-sided', zero_method='wilcox')
            out['wilcoxon_p_two_sided'] = float(w_two.pvalue)
    except Exception:
        pass

    return out


def run_top_vs_second_statistical_tests(
    summary_df: pd.DataFrame,
    save_dir: str,
    metrics_to_test: Sequence[str],
    seed: int,
    n_permutations: int,
    alpha: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare the best payment group with the second-best payment group.

    Input is results_target_long.csv-style data with one row per
    (payment_group, repeat, fold). For each metric, the function ranks groups
    by mean, aligns top and second-best groups by repeat/fold, and tests whether
    top - second > 0.
    """
    ensure_dir(save_dir)
    if summary_df.empty:
        raise ValueError('No rows found for statistical testing.')
    if 'payment_group' not in summary_df.columns:
        raise ValueError('summary_df must contain a payment_group column.')
    if 'repeat' not in summary_df.columns or 'fold' not in summary_df.columns:
        raise ValueError('summary_df must contain repeat and fold columns for paired statistical testing.')

    ranking_rows = []
    test_rows = []

    for metric_name in metrics_to_test:
        if metric_name not in summary_df.columns:
            print('Warning: metric {} not found in results. Skipping.'.format(metric_name), flush=True)
            continue

        keep_cols = ['payment_group', 'repeat', 'fold', metric_name]
        if 'target_company_id' in summary_df.columns:
            keep_cols.insert(0, 'target_company_id')
        metric_df = summary_df[keep_cols].copy()

        group_stats = (
            metric_df
            .groupby('payment_group')[metric_name]
            .agg(['mean', 'std', 'count'])
            .sort_values('mean', ascending=False)
            .reset_index()
        )
        group_stats.insert(0, 'metric', metric_name)
        group_stats['rank_descending'] = np.arange(1, len(group_stats) + 1)
        ranking_rows.append(group_stats)

        if group_stats.shape[0] < 2:
            print('Warning: fewer than two groups for metric {}. Skipping top-vs-second test.'.format(metric_name), flush=True)
            continue

        top_group = str(group_stats.iloc[0]['payment_group'])
        second_group = str(group_stats.iloc[1]['payment_group'])

        pivot_index = ['repeat', 'fold']
        if 'target_company_id' in metric_df.columns:
            pivot_index = ['target_company_id', 'repeat', 'fold']

        paired = metric_df.pivot_table(
            index=pivot_index,
            columns='payment_group',
            values=metric_name,
            aggfunc='mean',
        )
        if top_group not in paired.columns or second_group not in paired.columns:
            continue
        paired = paired[[top_group, second_group]].dropna()

        x = paired[top_group].to_numpy(dtype=float)
        y = paired[second_group].to_numpy(dtype=float)
        diffs = x - y
        n = int(diffs.size)

        if n == 0:
            test_rows.append({
                'metric': metric_name,
                'top_group': top_group,
                'second_group': second_group,
                'n_paired': 0,
                'note': 'No matched repeat/fold observations between top and second groups.',
            })
            continue

        perm = paired_sign_flip_permutation_test(
            x=x,
            y=y,
            n_permutations=n_permutations,
            seed=make_reproducible_seed('top_vs_second', seed, metric_name, top_group, second_group),
        )
        scipy_tests = scipy_paired_tests(x, y)

        std_diff = float(np.std(diffs, ddof=1)) if n > 1 else np.nan
        mean_diff = float(np.mean(diffs))
        se_diff = float(std_diff / np.sqrt(n)) if n > 1 and std_diff > 0 else np.nan
        ci_low = np.nan
        ci_high = np.nan
        if np.isfinite(se_diff):
            try:
                from scipy import stats
                critical = stats.t.ppf(0.975, df=n - 1)
                ci_low = float(mean_diff - critical * se_diff)
                ci_high = float(mean_diff + critical * se_diff)
            except Exception:
                ci_low = float(mean_diff - 1.96 * se_diff)
                ci_high = float(mean_diff + 1.96 * se_diff)

        row = {
            'metric': metric_name,
            'top_group': top_group,
            'second_group': second_group,
            'top_mean': float(np.mean(x)),
            'top_std': float(np.std(x, ddof=1)) if n > 1 else np.nan,
            'second_mean': float(np.mean(y)),
            'second_std': float(np.std(y, ddof=1)) if n > 1 else np.nan,
            'mean_diff_top_minus_second': mean_diff,
            'std_diff': std_diff,
            'se_diff': se_diff,
            'ci95_diff_low': ci_low,
            'ci95_diff_high': ci_high,
            'cohens_dz': float(mean_diff / std_diff) if std_diff and np.isfinite(std_diff) and std_diff > 0 else np.nan,
            'n_paired': n,
            'wins_top_gt_second': int(np.sum(diffs > 0)),
            'ties': int(np.sum(diffs == 0)),
            'losses_top_lt_second': int(np.sum(diffs < 0)),
            'permutation_p_one_sided': perm['permutation_p_one_sided'],
            'permutation_p_two_sided': perm['permutation_p_two_sided'],
            'paired_t_stat': scipy_tests['paired_t_stat'],
            'paired_t_p_one_sided': scipy_tests['paired_t_p_one_sided'],
            'paired_t_p_two_sided': scipy_tests['paired_t_p_two_sided'],
            'wilcoxon_stat': scipy_tests['wilcoxon_stat'],
            'wilcoxon_p_one_sided': scipy_tests['wilcoxon_p_one_sided'],
            'wilcoxon_p_two_sided': scipy_tests['wilcoxon_p_two_sided'],
            'alpha': float(alpha),
            'significant_by_permutation_one_sided': bool(perm['permutation_p_one_sided'] < alpha) if np.isfinite(perm['permutation_p_one_sided']) else False,
            'significant_by_paired_t_one_sided': bool(scipy_tests['paired_t_p_one_sided'] < alpha) if np.isfinite(scipy_tests['paired_t_p_one_sided']) else False,
            'significant_by_wilcoxon_one_sided': bool(scipy_tests['wilcoxon_p_one_sided'] < alpha) if np.isfinite(scipy_tests['wilcoxon_p_one_sided']) else False,
            'note': 'Primary test: paired sign-flip permutation, one-sided alternative top > second. Pairing key: repeat/fold.',
        }
        test_rows.append(row)

    ranking_df = pd.concat(ranking_rows, axis=0, ignore_index=True) if ranking_rows else pd.DataFrame()
    tests_df = pd.DataFrame(test_rows)

    ranking_path = os.path.join(save_dir, 'results_group_ranking.csv')
    tests_path = os.path.join(save_dir, 'statistical_tests_top_vs_second.csv')
    ranking_df.to_csv(ranking_path, index=False)
    tests_df.to_csv(tests_path, index=False)

    if not tests_df.empty:
        print('\nStatistical tests: top group vs. second-best group')
        display_cols = [
            'metric',
            'top_group',
            'second_group',
            'mean_diff_top_minus_second',
            'n_paired',
            'permutation_p_one_sided',
            'significant_by_permutation_one_sided',
        ]
        print(tests_df[display_cols].to_string(index=False))
        print('Saved statistical tests to:', tests_path, flush=True)
    else:
        print('\nNo statistical tests were produced. Check metrics and group availability.', flush=True)

    return ranking_df, tests_df

# -----------------------------------------------------------------------------
# Main experiment loop
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Payment-level auxiliary advertiser experiment.')
    parser.add_argument('--data_path', type=str, default='data.pickle')
    parser.add_argument('--save_dirs', type=str, default='results_payment_level_auxiliary_stat_107800070')

    parser.add_argument('--target_company_id', type=int, default=107800070 ,
                        help='The target company whose test performance will be compared.')
    parser.add_argument('--payment_groups', type=str, default='all',
                        help='Payment groups to run: all, or comma-separated low_payment,mid_payment,high_payment.')
    parser.add_argument('--num_aux', type=int, default=2,
                        help='Number of auxiliary companies sampled from each payment group.')
    parser.add_argument('--sample_repeats', type=int, default=5,
                        help='Number of auxiliary-company samples per payment group.')
    parser.add_argument('--fold', type=int, default=5)
    parser.add_argument('--test_size', type=float, default=0.2)

    parser.add_argument('--num_experts', type=int, default=5)
    parser.add_argument('--epoch_max', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--lr_patience', type=int, default=5)
    parser.add_argument('--min_lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--eval_batch_size', type=int, default=65536)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=45)
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers. On Windows, 0 is the safest choice and often prevents 0xC0000005 crashes.')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='Number of batches prefetched by each DataLoader worker when num_workers > 0.')
    parser.add_argument('--pin_memory', action='store_true',
                        help='Enable CUDA pinned-memory DataLoader transfer. Faster on some systems, but disable it if Windows crashes.')
    parser.add_argument('--resume', action='store_true',
                        help='Skip payment_group/repeat/fold runs already present in results_target_long.csv.')
    parser.add_argument('--empty_cache', dest='empty_cache', action='store_true', default=True,
                        help='Call torch.cuda.empty_cache() after each finished run. Enabled by default.')
    parser.add_argument('--no_empty_cache', dest='empty_cache', action='store_false',
                        help='Disable torch.cuda.empty_cache() after each finished run.')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--amp', action='store_true',
                        help='Use CUDA mixed precision to speed up training. It may cause tiny numerical differences.')
    parser.add_argument('--strict_deterministic', action='store_true',
                        help='Use stricter PyTorch deterministic algorithms. May raise errors for some CUDA ops.')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--skip_stat_tests', action='store_true',
                        help='Do not run the top-vs-second statistical tests after training.')
    parser.add_argument('--stats_only', action='store_true',
                        help='Only run statistical tests using an existing results_target_long.csv; no training is performed.')
    parser.add_argument('--stat_test_metrics', type=str, default='target_AUCROC,target_GINI',
                        help='Comma-separated metrics to test, for example target_AUCROC,target_GINI.')
    parser.add_argument('--stat_test_permutations', type=int, default=10000,
                        help='Number of random sign-flip permutations for the paired permutation test.')
    parser.add_argument('--stat_test_alpha', type=float, default=0.05,
                        help='Significance level for top-vs-second tests.')

    args = parser.parse_args()
    set_seed(args.seed, strict_deterministic=args.strict_deterministic)

    ensure_dir(args.save_dirs)
    samples_path = os.path.join(args.save_dirs, 'auxiliary_samples.csv')
    target_long_path = os.path.join(args.save_dirs, 'results_target_long.csv')
    all_tasks_long_path = os.path.join(args.save_dirs, 'results_all_tasks_long.csv')
    aggregate_path = os.path.join(args.save_dirs, 'results_target_aggregate.csv')

    if args.stats_only:
        if not os.path.exists(target_long_path):
            raise FileNotFoundError('Cannot find {}. Run training first or set --save_dirs correctly.'.format(target_long_path))
        summary_df = pd.read_csv(target_long_path)
        run_top_vs_second_statistical_tests(
            summary_df=summary_df,
            save_dir=args.save_dirs,
            metrics_to_test=parse_metric_list(args.stat_test_metrics),
            seed=args.seed,
            n_permutations=args.stat_test_permutations,
            alpha=args.stat_test_alpha,
        )
        print('Stats-only mode finished. Results saved to:', args.save_dirs, flush=True)
        return

    if args.device == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but torch.cuda.is_available() is False. Check your PyTorch/CUDA installation.')
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    with open(args.data_path, 'rb') as file:
        # Use pandas read_pickle for compatibility with pandas-created pickle files.
        data_dict = pd.read_pickle(args.data_path)

    data_dict = {int(k): v for k, v in data_dict.items()}
    all_company_ids = sorted(list(data_dict.keys()))

    if args.target_company_id not in all_company_ids:
        raise ValueError('target_company_id {} not found. Available companies: {}'.format(
            args.target_company_id, all_company_ids
        ))

    payment_groups_to_run = parse_group_list(args.payment_groups)

    ensure_dir(args.save_dirs)

    payment_summary = compute_company_payment_summary(data_dict)
    payment_summary = assign_payment_groups(payment_summary)
    payment_summary.to_csv(os.path.join(args.save_dirs, 'company_payment_groups.csv'), index=False)

    print('Device:', device)
    if device.type == 'cuda':
        print('CUDA device name:', torch.cuda.get_device_name(device))
        print('AMP enabled:', bool(args.amp))
        print('DataLoader pin_memory:', bool(args.pin_memory and device.type == 'cuda'))
        print('DataLoader num_workers:', args.num_workers)
    print('All companies:', all_company_ids)
    print('Target company:', args.target_company_id)
    print('Payment groups to run:', payment_groups_to_run)
    print('\nCompany payment grouping:')
    print(payment_summary[['company_id', 'mean_user_payment', 'payment_group']].to_string(index=False))

    train_val_ids_fold, test_ids = make_fixed_splits(
        data_dict=data_dict,
        all_company_ids=all_company_ids,
        folds=args.fold,
        seed=args.seed,
        test_size=args.test_size,
    )

    encoders, _ = fit_global_encoders(data_dict, all_company_ids)

    sample_rows = []
    summary_rows = []
    all_task_rows = []

    samples_path = os.path.join(args.save_dirs, 'auxiliary_samples.csv')
    target_long_path = os.path.join(args.save_dirs, 'results_target_long.csv')
    all_tasks_long_path = os.path.join(args.save_dirs, 'results_all_tasks_long.csv')
    aggregate_path = os.path.join(args.save_dirs, 'results_target_aggregate.csv')

    completed_keys = set()
    if args.resume:
        if os.path.exists(samples_path):
            existing_samples = pd.read_csv(samples_path)
            sample_rows = existing_samples.to_dict('records')
        if os.path.exists(target_long_path):
            existing_summary = pd.read_csv(target_long_path)
            summary_rows = existing_summary.to_dict('records')
            for _, row in existing_summary.iterrows():
                completed_keys.add((str(row['payment_group']), int(row['repeat']), int(row['fold'])))
            print('Resume enabled. Found {} completed target-result rows.'.format(len(completed_keys)), flush=True)
        if os.path.exists(all_tasks_long_path):
            all_task_rows.append(pd.read_csv(all_tasks_long_path))

    for payment_group in payment_groups_to_run:
        group_company_ids = payment_summary.loc[
            payment_summary['payment_group'] == payment_group, 'company_id'
        ].astype(int).tolist()

        auxiliary_sets = make_auxiliary_sets_for_group(
            group_company_ids=group_company_ids,
            target_company_id=args.target_company_id,
            num_aux=args.num_aux,
            repeats=args.sample_repeats,
            seed=args.seed,
            group_name=payment_group,
        )

        for repeat, auxiliary_company_ids in enumerate(auxiliary_sets):
            selected_company_ids = [args.target_company_id] + [cid for cid in auxiliary_company_ids if cid != args.target_company_id]

            sample_rows.append({
                'target_company_id': args.target_company_id,
                'payment_group': payment_group,
                'repeat': repeat,
                'auxiliary_company_ids': '|'.join(str(cid) for cid in auxiliary_company_ids),
                'selected_company_ids': '|'.join(str(cid) for cid in selected_company_ids),
            })

            for fold_idx, (train_ids, val_ids) in enumerate(train_val_ids_fold):
                run_name = 'target_{}_group_{}_repeat_{}_fold_{}'.format(
                    args.target_company_id, payment_group, repeat, fold_idx
                )
                run_save_path = os.path.join(args.save_dirs, run_name)
                completed_key = (payment_group, repeat, fold_idx)
                if args.resume and completed_key in completed_keys:
                    print('Skipping completed {}'.format(run_name), flush=True)
                    continue

                run_seed = make_reproducible_seed(
                    'train_run',
                    args.seed,
                    args.target_company_id,
                    payment_group,
                    repeat,
                    fold_idx,
                    args.num_aux,
                )
                set_seed(run_seed, strict_deterministic=args.strict_deterministic)

                print('\nRunning {}, selected_companies={}'.format(run_name, selected_company_ids), flush=True)

                target_test, all_task_test, _ = train_one_run(
                    args=args,
                    data_dict=data_dict,
                    encoders=encoders,
                    train_ids=train_ids,
                    val_ids=val_ids,
                    test_ids=test_ids,
                    selected_company_ids=selected_company_ids,
                    target_company_id=args.target_company_id,
                    run_save_path=run_save_path,
                    device=device,
                )

                summary = {
                    'target_company_id': args.target_company_id,
                    'payment_group': payment_group,
                    'repeat': repeat,
                    'fold': fold_idx,
                    'auxiliary_company_ids': '|'.join(str(cid) for cid in auxiliary_company_ids),
                    'selected_company_ids': '|'.join(str(cid) for cid in selected_company_ids),
                    'target_AUCROC': target_test['AUCROC'],
                    'target_GINI': target_test['GINI'],
                    'target_test_loss': target_test['loss'],
                }
                summary_rows.append(summary)

                all_task_test = all_task_test.copy()
                all_task_test['target_company_id'] = args.target_company_id
                all_task_test['payment_group'] = payment_group
                all_task_test['repeat'] = repeat
                all_task_test['fold'] = fold_idx
                all_task_test['auxiliary_company_ids'] = '|'.join(str(cid) for cid in auxiliary_company_ids)
                all_task_test['selected_company_ids'] = '|'.join(str(cid) for cid in selected_company_ids)
                all_task_rows.append(all_task_test)

                completed_keys.add(completed_key)

                # Save after every run so partial results are preserved.
                pd.DataFrame(sample_rows).drop_duplicates().to_csv(samples_path, index=False)

                summary_df = pd.DataFrame(summary_rows)
                summary_df = summary_df.drop_duplicates(
                    subset=['target_company_id', 'payment_group', 'repeat', 'fold'],
                    keep='last',
                )
                summary_df.to_csv(target_long_path, index=False)

                all_tasks_df = pd.concat(all_task_rows, axis=0, ignore_index=True)
                all_tasks_df = all_tasks_df.drop_duplicates(
                    subset=['target_company_id', 'payment_group', 'repeat', 'fold', 'company_id'],
                    keep='last',
                )
                all_tasks_df.to_csv(all_tasks_long_path, index=False)

                aggregate_df = (
                    summary_df
                    .groupby(['target_company_id', 'payment_group'])[['target_AUCROC', 'target_GINI', 'target_test_loss']]
                    .agg(['mean', 'std', 'count'])
                    .reset_index()
                )
                aggregate_df.columns = [
                    '_'.join(col).strip('_') if isinstance(col, tuple) else col
                    for col in aggregate_df.columns
                ]
                aggregate_df.to_csv(aggregate_path, index=False)

                if device.type == 'cuda' and args.empty_cache:
                    torch.cuda.empty_cache()

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).drop_duplicates(
            subset=['target_company_id', 'payment_group', 'repeat', 'fold'],
            keep='last',
        )
        aggregate_df = (
            summary_df
            .groupby(['target_company_id', 'payment_group'])[['target_AUCROC', 'target_GINI', 'target_test_loss']]
            .agg(['mean', 'std', 'count'])
            .reset_index()
        )
        aggregate_df.columns = [
            '_'.join(col).strip('_') if isinstance(col, tuple) else col
            for col in aggregate_df.columns
        ]
        summary_df.to_csv(target_long_path, index=False)
        aggregate_df.to_csv(aggregate_path, index=False)

        if not args.skip_stat_tests:
            run_top_vs_second_statistical_tests(
                summary_df=summary_df,
                save_dir=args.save_dirs,
                metrics_to_test=parse_metric_list(args.stat_test_metrics),
                seed=args.seed,
                n_permutations=args.stat_test_permutations,
                alpha=args.stat_test_alpha,
            )

    print('\nFinished. Results saved to:', args.save_dirs, flush=True)
    print('Main files:')
    print('  - company_payment_groups.csv')
    print('  - auxiliary_samples.csv')
    print('  - results_target_long.csv')
    print('  - results_target_aggregate.csv')
    print('  - results_all_tasks_long.csv')
    print('  - results_group_ranking.csv')
    print('  - statistical_tests_top_vs_second.csv')


if __name__ == '__main__':
    main()
