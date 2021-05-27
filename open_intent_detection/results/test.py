import pandas as pd 
from datetime import datetime
import os
df = pd.read_csv('results_final.csv')
cols = ['dataset', 'method',  'Acc', 'F1', 'F1-open', 'F1-known', 'known_cls_ratio', 'labeled_ratio','seed']
df_mean = df[cols].groupby(['method', 'dataset', 'known_cls_ratio', 'labeled_ratio']).mean().round(2)

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
results_log_dir = 'logs/'
results_file_name = TIMESTAMP + '.csv'


if not os.path.exists(results_log_dir):
    os.makedirs(results_log_dir)

results_path = os.path.join(results_log_dir, results_file_name)
df_mean.to_csv(results_path,index=False)