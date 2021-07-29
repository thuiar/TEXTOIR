import pandas as pd
import os
import argparse
#read A, compute mean

def results_mean_csv(input_dir, input_file_name, output_dir, output_file_name = None):

    if output_file_name == None:
        output_file_name = input_file_name

    df = pd.read_csv(os.path.join(input_dir, input_file_name))
    cols = ['dataset', 'method',  'ACC', 'ARI', 'NMI', 'seed']

    df_mean = df[cols].groupby(['dataset', 'method'], as_index=False).mean().round(2)
    df_mean = df_mean.groupby(['dataset']).apply(lambda x : x.sort_values(['method']))
    df_mean['id'] = df_mean['dataset'] + '-' + df_mean['method']
    df_mean = df_mean.reindex(columns=['id', 'ACC', 'ARI', 'NMI', 'seed'])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results_path = os.path.join(output_dir, output_file_name)
    df_mean.to_csv(results_path,index=False)

def compute_gap(old_file, new_file, gap_dir, input_file_name):

    df_new = pd.read_csv(new_file, index_col='id')
    df_old = pd.read_csv(old_file, index_col='id')
    
    common_ids = set(df_old.index) & set(df_new.index)
    if common_ids is not None:
        df_old = df_old.loc[common_ids]
        df_res = (df_new[['ACC', 'ARI', 'NMI']] - df_old[['ACC', 'ARI', 'NMI']]).round(2).loc[common_ids].sort_index()
        results_path = os.path.join(gap_dir, input_file_name)
        df_res.to_csv(results_path,index=True)
    else:
        print("error: Two files must have data with the same id")

if __name__ == '__main__':        

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_name', type=str, default='KM.csv')
    args = parser.parse_args()
    # input_file_name = args.method + '.csv'
    input_dir = 'mean/'
    gap_dir = 'gap/'
    results_mean_csv("", 'original.csv',"gap", 'standard.csv')
    results_mean_csv(input_dir, args.input_file_name, gap_dir)
    new_file = os.path.join(gap_dir, args.input_file_name)
    old_file = os.path.join(gap_dir, 'standard.csv')

    compute_gap(old_file, new_file, gap_dir, args.input_file_name)






