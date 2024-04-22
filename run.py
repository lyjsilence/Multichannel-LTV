import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import utils
import pickle
from models import cross_ltv, single_ltv
import random
from training import Trainer
import warnings
warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


parser = argparse.ArgumentParser(description="Cross-trained LTV for Kaggle Acquire Valued Shoppers Challenge dataset")
parser.add_argument('--save_dirs', type=str, default='results', help='The dirs for saving results')
parser.add_argument('--log', type=bool, default=True, help='Whether log the information of training process')

parser.add_argument('--model_name', type=str, default='cross_ltv', choices=['single_ltv', 'cross_ltv'],
                    help='The model want to implement')
parser.add_argument('--company_id', type=int, default=10000, help='The company want to be trained',
                    choices=[10000, 101200010, 101410010, 101600010, 102100020, 102700020, 102840020, 103000030,
                             103338333, 103400030, 103600030, 103700030, 103800030, 104300040, 104400040, 104470040,
                             104900040, 105100050, 105150050, 107800070])

parser.add_argument('--num_experts', type=int, default=5, help='The number of expert')
parser.add_argument('--fold', type=int, default=5, help='The number of fold for cross validation')
parser.add_argument('--epoch_max', type=int, default=50, help='The number of epoch for one experiment')
parser.add_argument('--lr', type=float, default=0.003, help='The learning rate when training NN')
parser.add_argument('--batch_size', type=int, default=1024, help='The batch size when training NN')
parser.add_argument('--weight_decay', type=float, default=0, help='The weight decay when training NN')
parser.add_argument('--dropout', type=float, default=0.5)

args = parser.parse_args()


if __name__ == '__main__':
    set_seed()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    save_path = os.path.join(args.save_dirs, args.model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open('data.pickle', 'rb') as file:
        data_dict = pickle.load(file)

    # train, val, test split
    train_val_ids_fold, test_ids = utils.train_val_test_split(data_dict, folds=5)

    if args.model_name == 'single_ltv':
        data = data_dict[args.company_id]
        # preprocess the data and encode the categorical features
        data, feat_sizes = utils.preprocess_single(data)
    else:
        # preprocess the data and encode the categorical features
        data, feat_sizes = utils.preprocess_cross(data_dict)

    for exp_id in range(args.fold):
        train_ids, val_ids = train_val_ids_fold[exp_id][0], train_val_ids_fold[exp_id][1]

        if args.model_name == 'single_ltv':
            ltv_dataset = utils.single_ltv_dataset
            ltv_collate_fn = utils.single_ltv_collate_fn
        else:
            ltv_dataset = utils.cross_ltv_dataset
            ltv_collate_fn = utils.cross_ltv_collate_fn

        train_data = ltv_dataset(args, data, train_ids)
        val_data = ltv_dataset(args, data, val_ids)
        test_data = ltv_dataset(args, data, test_ids)

        dl_train = DataLoader(dataset=train_data, collate_fn=ltv_collate_fn,
                              shuffle=True, batch_size=args.batch_size, num_workers=1, pin_memory=False)
        dl_val = DataLoader(dataset=val_data, collate_fn=ltv_collate_fn,
                            shuffle=False, batch_size=len(val_data), num_workers=1, pin_memory=False)
        dl_test = DataLoader(dataset=test_data, collate_fn=ltv_collate_fn,
                             shuffle=False, batch_size=len(test_data), num_workers=1, pin_memory=False)

        if args.model_name == 'cross_ltv':
            model = cross_ltv(args, feat_sizes, device).to(device)
        elif args.model_name == 'single_ltv':
            model = single_ltv(args, feat_sizes, device).to(device)
        else:
            ModuleNotFoundError(f'Module {args.model_name} not found')

        print(f'Training model: {args.model_name}, Experiment: {exp_id}, '
              f'Num of training parameters: {sum(p.numel() for p in model.parameters())}')
        Trainer(model, dl_train, dl_val, dl_test, args, device, exp_id)