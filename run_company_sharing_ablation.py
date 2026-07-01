"""
Company-sharing ablation experiment for the Cross-Learning LTV project.

Purpose
-------
This script answers the reviewer question: does performance change when fewer
advertisers participate in the cross-advertiser/shared multi-task training?

Protocol
--------
For a fixed target company, keep the same train/validation/test users and train
cross-LTV models with different numbers of shared companies:

    share_size =2, 5, 10, 15

For each share_size m, the target company is always included and m-1 auxiliary
companies are sampled from the remaining companies. Performance is reported for
THE SAME target company and THE SAME test users across all share sizes. This
isolates the effect of cross-advertiser data sharing from ordinary train/test
variation.

Example
-------
python run_company_sharing_ablation.py \
    --data_path data.pickle \
    --target_company_ids 101600010,103000030\
    --share_sizes 2,5,10,15 \
    --sample_repeats 5 \
    --fold 5 \
    --epoch_max 50

For a paper-level aggregate over all advertisers, use:

python run_company_sharing_ablation.py --target_company_ids all
"""

import argparse
import copy
import os
import pickle
import random
import time
from typing import Dict, Iterable, List, Sequence, Tuple

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


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def parse_int_list(value: str) -> List[int]:
    """Parse a comma-separated integer list."""
    return [int(x.strip()) for x in value.split(',') if x.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


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
    """Create one fixed user split reused by every sharing condition.

    The split is based on users that appear in all companies, so every sampled
    company subset is evaluated on exactly the same users.
    """
    user_ids = get_common_user_ids(data_dict, all_company_ids)
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


def fit_global_encoders(
    data_dict: Dict[int, pd.DataFrame],
    company_ids: Sequence[int],
) -> Tuple[Dict[str, LabelEncoder], Dict[str, List[int]]]:
    """Fit categorical encoders once, usually on all 20 advertisers.

    Using the same encoders across all share_size values avoids changing the
    categorical vocabulary merely because fewer auxiliary companies are present.
    """
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
            # The encoders are fitted on all companies by default. Cast to str so
            # train/test and old pandas categorical dtypes are handled consistently.
            df[feature] = encoders[feature].transform(df[feature].astype(str))
        encoded[cid] = df

    return encoded, feat_sizes


def _validate_company_subset_request(
    all_company_ids: Sequence[int],
    target_company_id: int,
    share_size: int,
) -> None:
    if share_size < 1:
        raise ValueError('share_size must be >= 1')
    if share_size > len(all_company_ids):
        raise ValueError('share_size cannot exceed the number of available companies')
    if target_company_id not in all_company_ids:
        raise ValueError('target_company_id is not in data_dict')


def sample_company_subset(
    all_company_ids: Sequence[int],
    target_company_id: int,
    share_size: int,
    repeat: int,
    seed: int,
) -> List[int]:
    """Randomly sample one company subset that always contains the target company.

    This function is kept for the optional ``--aux_sampling random`` setting.
    It is reproducible for a fixed target, share_size, repeat, and seed, but it
    samples repeats independently, so the same auxiliary company may appear
    frequently across repeats by chance.
    """
    _validate_company_subset_request(all_company_ids, target_company_id, share_size)

    if share_size == len(all_company_ids):
        selected = list(all_company_ids)
    elif share_size == 1:
        selected = [target_company_id]
    else:
        rng = np.random.RandomState(seed + 1009 * repeat + 37 * share_size + target_company_id % 997)
        candidates = [cid for cid in all_company_ids if cid != target_company_id]
        auxiliaries = rng.choice(candidates, size=share_size - 1, replace=False).tolist()
        selected = [target_company_id] + auxiliaries

    # Put the target first. This makes target_idx = 0 and simplifies reporting.
    selected = [target_company_id] + [cid for cid in selected if cid != target_company_id]
    return selected


def make_balanced_company_subsets(
    all_company_ids: Sequence[int],
    target_company_id: int,
    share_size: int,
    repeats: int,
    seed: int,
) -> List[List[int]]:
    """Create repeat-level company subsets with balanced auxiliary coverage.

    The target company is included in every subset. For auxiliary companies,
    each repeat chooses the companies that have been used least often so far,
    with seeded random tie-breaking.

    Consequences:
    - For share_size=2 and repeats<=19, no auxiliary company is repeated.
    - More generally, auxiliary exposure counts differ by at most one whenever
      possible, so repeats do not accidentally focus on the same few companies.
    - The output is deterministic for a fixed target, share_size, repeats, and
      seed.
    """
    _validate_company_subset_request(all_company_ids, target_company_id, share_size)

    if repeats < 1:
        raise ValueError('repeats must be >= 1')

    if share_size == 1:
        return [[target_company_id] for _ in range(repeats)]
    if share_size == len(all_company_ids):
        return [list(all_company_ids) for _ in range(repeats)]

    candidates = [cid for cid in all_company_ids if cid != target_company_id]
    aux_per_repeat = share_size - 1
    rng = np.random.RandomState(seed + 7919 * share_size + target_company_id % 1009)

    exposure_count = {cid: 0 for cid in candidates}
    subsets = []
    used_subsets = set()

    for repeat in range(repeats):
        # Seeded random tie-breaking among companies with the same exposure count.
        tie_breaker = {cid: rng.rand() for cid in candidates}
        ranked = sorted(candidates, key=lambda cid: (exposure_count[cid], tie_breaker[cid]))
        aux = ranked[:aux_per_repeat]

        subset_key = tuple(sorted(aux))
        if subset_key in used_subsets:
            # Preserve balanced exposure as the primary objective. If an identical
            # combination appears, try to swap one company with another company
            # from the same or next-lowest exposure level. This is deterministic
            # because it uses the same seeded RNG.
            aux_set = set(aux)
            best_max_count = max(exposure_count[cid] for cid in aux)
            swap_pool = [
                cid for cid in ranked[aux_per_repeat:]
                if exposure_count[cid] <= best_max_count + 1
            ]
            rng.shuffle(swap_pool)
            replaced = False
            for candidate in swap_pool:
                for old in sorted(aux, key=lambda cid: (-exposure_count[cid], tie_breaker[cid])):
                    trial = aux.copy()
                    trial[trial.index(old)] = candidate
                    trial_key = tuple(sorted(trial))
                    if trial_key not in used_subsets and len(set(trial)) == len(trial):
                        aux = trial
                        subset_key = trial_key
                        replaced = True
                        break
                if replaced:
                    break

        for cid in aux:
            exposure_count[cid] += 1

        used_subsets.add(subset_key)
        subsets.append([target_company_id] + aux)

    return subsets

def make_company_subsets(
    all_company_ids: Sequence[int],
    target_company_id: int,
    share_size: int,
    repeats: int,
    seed: int,
    strategy: str,
) -> List[List[int]]:
    """Return all repeat-level subsets for one target/share_size condition."""
    if strategy == 'balanced':
        return make_balanced_company_subsets(
            all_company_ids=all_company_ids,
            target_company_id=target_company_id,
            share_size=share_size,
            repeats=repeats,
            seed=seed,
        )
    if strategy == 'random':
        return [
            sample_company_subset(
                all_company_ids=all_company_ids,
                target_company_id=target_company_id,
                share_size=share_size,
                repeat=repeat,
                seed=seed,
            )
            for repeat in range(repeats)
        ]
    raise ValueError('Unknown auxiliary sampling strategy: {}'.format(strategy))


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
        self.company_ids = list(company_ids)
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
    company_ids = list(company_ids)

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
            x_cat = x_cat.to(self.device)
            embeddings = [self.embeddings[i](x_cat[:, task_idx]) for task_idx in range(x_cat.shape[1])]
            embeddings_list.append(torch.cat(embeddings, dim=1))

        x_cat = torch.cat(embeddings_list, dim=1)
        x_num = self.numeric(x_num.to(self.device))
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
            # The AUC calculation still applies sigmoid to `prob` as in utils.py.
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
    normalized = raw / raw.iloc[0]
    return pd.DataFrame({'raw': raw, 'normalized': normalized})[['raw', 'normalized']]


def compute_auc_and_gini(y_true: np.ndarray, prob_logits: np.ndarray, ltv_pred: np.ndarray) -> Tuple[float, float]:
    positive = (y_true > 0).astype(float)
    prob = 1.0 / (1.0 + np.exp(-prob_logits))

    # AUC is undefined if the test fold has only one class. This should not
    # happen in the full dataset, but returning NaN keeps batch runs alive.
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
        rows.append({'company_id': cid, 'task_idx': task_idx, 'AUCROC': auc, 'GINI': gini})

    all_task_df = pd.DataFrame(rows)
    target_row = all_task_df[all_task_df['company_id'] == target_company_id].iloc[0].to_dict()
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

    dl_train = DataLoader(
        dataset=train_dataset,
        collate_fn=collate_fn,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    dl_val = DataLoader(
        dataset=val_dataset,
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    dl_test = DataLoader(
        dataset=test_dataset,
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    args.num_tasks = len(selected_company_ids)
    model = CrossLTVDynamic(args, feat_sizes, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.lr_patience, min_lr=args.min_lr
    )

    ensure_dir(run_save_path)
    best_path = os.path.join(run_save_path, 'cross_ltv_dynamic_best.pkl')
    best_val_loss = np.inf
    log_rows = []

    for epoch in range(1, args.epoch_max + 1):
        model.train()
        epoch_start = time.time()
        train_class_losses = []
        train_reg_losses = []

        for batch_idx, batch in enumerate(dl_train, start=1):
            x_all = model(batch['x_cat'], batch['x_num'])
            class_loss, reg_loss = model.compute_loss(x_all, batch['y'])
            loss = class_loss + reg_loss

            optimizer.zero_grad()
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
# Main experiment loop
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Ablation over the number of shared advertisers.')
    parser.add_argument('--data_path', type=str, default='data.pickle')
    parser.add_argument('--save_dirs', type=str, default='results_company_sharing_ablation')

    parser.add_argument('--target_company_ids', type=str, default='103000030,101600010',
                        help='Comma-separated target company ids, or "all".')
    parser.add_argument('--share_sizes', type=str, default='2,5,10,15',
                        help='Comma-separated numbers of companies used in shared training.')
    parser.add_argument('--sample_repeats', type=int, default=5,
                        help='Number of auxiliary-company subsets per share size.')
    parser.add_argument('--aux_sampling', type=str, default='balanced', choices=['balanced', 'random'],
                        help='How to choose auxiliary companies across repeats. Balanced reduces repeated auxiliaries.')
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
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    set_seed(args.seed)

    if args.device == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    with open(args.data_path, 'rb') as file:
        # data_dict = pickle.load(file)
        data_dict = pd.read_pickle("data.pickle")


    # Normalize keys to int in case pickle stores numpy integer keys.
    data_dict = {int(k): v for k, v in data_dict.items()}
    all_company_ids = sorted(list(data_dict.keys()))

    if args.target_company_ids.lower() == 'all':
        target_company_ids = all_company_ids
    else:
        target_company_ids = parse_int_list(args.target_company_ids)

    share_sizes = parse_int_list(args.share_sizes)
    for share_size in share_sizes:
        if share_size > len(all_company_ids):
            raise ValueError('share_size={} exceeds number of companies={}'.format(share_size, len(all_company_ids)))

    ensure_dir(args.save_dirs)

    print('Device:', device)
    print('All companies:', all_company_ids)
    print('Target companies:', target_company_ids)
    print('Share sizes:', share_sizes)

    train_val_ids_fold, test_ids = make_fixed_splits(
        data_dict=data_dict,
        all_company_ids=all_company_ids,
        folds=args.fold,
        seed=args.seed,
        test_size=args.test_size,
    )

    encoders, _ = fit_global_encoders(data_dict, all_company_ids)

    summary_rows = []
    all_task_rows = []

    for target_company_id in target_company_ids:
        if target_company_id not in all_company_ids:
            raise ValueError('target company {} not found in data_dict'.format(target_company_id))

        for share_size in share_sizes:
            repeats = 1 if share_size in [1, len(all_company_ids)] else args.sample_repeats
            company_subsets = make_company_subsets(
                all_company_ids=all_company_ids,
                target_company_id=target_company_id,
                share_size=share_size,
                repeats=repeats,
                seed=args.seed,
                strategy=args.aux_sampling,
            )

            for repeat, selected_company_ids in enumerate(company_subsets):
                for fold_idx, (train_ids, val_ids) in enumerate(train_val_ids_fold):
                    run_seed = args.seed + 100000 * fold_idx + 1000 * repeat + share_size
                    set_seed(run_seed)

                    run_name = 'target_{}_share_{}_repeat_{}_fold_{}'.format(
                        target_company_id, share_size, repeat, fold_idx
                    )
                    run_save_path = os.path.join(args.save_dirs, run_name)

                    print(
                        '\nRunning {}, selected_companies={}'.format(run_name, selected_company_ids)
                    )

                    target_test, all_task_test, _ = train_one_run(
                        args=args,
                        data_dict=data_dict,
                        encoders=encoders,
                        train_ids=train_ids,
                        val_ids=val_ids,
                        test_ids=test_ids,
                        selected_company_ids=selected_company_ids,
                        target_company_id=target_company_id,
                        run_save_path=run_save_path,
                        device=device,
                    )

                    summary = {
                        'target_company_id': target_company_id,
                        'share_size': share_size,
                        'repeat': repeat,
                        'fold': fold_idx,
                        'selected_company_ids': '|'.join(str(cid) for cid in selected_company_ids),
                        'target_AUCROC': target_test['AUCROC'],
                        'target_GINI': target_test['GINI'],
                        'target_test_loss': target_test['loss'],
                    }
                    summary_rows.append(summary)

                    all_task_test = all_task_test.copy()
                    all_task_test['target_company_id'] = target_company_id
                    all_task_test['share_size'] = share_size
                    all_task_test['repeat'] = repeat
                    all_task_test['fold'] = fold_idx
                    all_task_test['selected_company_ids'] = '|'.join(str(cid) for cid in selected_company_ids)
                    all_task_rows.append(all_task_test)

                    summary_df = pd.DataFrame(summary_rows)
                    summary_df.to_csv(os.path.join(args.save_dirs, 'results_target_long.csv'), index=False)

                    all_tasks_df = pd.concat(all_task_rows, axis=0, ignore_index=True)
                    all_tasks_df.to_csv(os.path.join(args.save_dirs, 'results_all_tasks_long.csv'), index=False)

                    aggregate_df = (
                        summary_df
                        .groupby(['target_company_id', 'share_size'])[['target_AUCROC', 'target_GINI']]
                        .agg(['mean', 'std', 'count'])
                    )
                    aggregate_df.to_csv(os.path.join(args.save_dirs, 'results_target_aggregate.csv'))

    print('\nFinished. Results saved to:', args.save_dirs)
    print('Main files:')
    print('  - results_target_long.csv')
    print('  - results_target_aggregate.csv')
    print('  - results_all_tasks_long.csv')


if __name__ == '__main__':
    main()
