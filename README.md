# MTL-VBB
 The User Lifetime Value Cross Learning for Kaggle Acquire Valued Shoppers Challenge dataset

### Requirements
Python == 3.8.   
Pytorch: 1.8.1+cu102, Sklearn:0.23.2, Numpy: 1.19.2, Pandas: 1.1.3, Matplotlib: 3.3.2   
All the codes are run on GPUs by default. 


### Preprocessing
The Kaggle Acquire Valued Shoppers Challenge dataset can be downloaded from https://www.kaggle.com/c/acquire-valued-shoppers-challenge. 

We follow the preprocessing code from https://github.com/google/lifetime_value/blob/master/notebooks/kaggle_acquire_valued_shoppers_challenge/preprocess_data.ipynb. 

Furthermore, we select consumers who have consumption records in all 20 companies, resulting in a total of 66,263 consumers. The dataset is stored in data.pickle. 

### Kaggle Acquire Valued Shoppers Challenge Experiments

Train the baseline models (treat each company independently)
```
python3 run.py --model_name single_ltv --company_id 10000 --num_exp 5 
```
The company_id can be replaced by 101200010, 101410010, 101600010, 102100020, 102700020, 102840020, 103000030, 103338333, 103400030, 103600030, 103700030, 103800030, 104300040, 104400040, 104470040, 104900040, 105100050, 105150050, 107800070

Train the cross-learning model
```
python3 run.py --model_name cross_ltv --num_exp 5 
```

These experiments generate Table EC.7 in the paper.

### Company-sharing ablation

Run the company-sharing ablation for Table EC.8:
```
python3 run_company_sharing_ablation.py --data_path data.pickle --target_company_ids all
```

### Payment-level auxiliary experiments

Run the payment-level auxiliary experiments for Table EC.9:
```
python3 run_payment_level_auxiliary_experiment_101600010.py --data_path data.pickle
python3 run_payment_level_auxiliary_experiment_103700030.py --data_path data.pickle
python3 run_payment_level_auxiliary_experiment_107800070.py --data_path data.pickle
```
