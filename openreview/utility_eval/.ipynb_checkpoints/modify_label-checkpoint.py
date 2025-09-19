import pandas as pd
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Modify column name to fit evaluation code.")
    parser.add_argument("--target_dir", help="Path to the directory.")
    parser.add_argument("--iteration", type=int, help="Which iteration.")
    args = parser.parse_args()
    print(args)
    
    target_dir = args.target_dir
    iteration = args.iteration
    syn_df = pd.read_csv(f"{target_dir}/PE_iter_{iteration}.csv")
    syn_df = syn_df.rename(columns={'area': 'label1','recommendation':'label2'})
    syn_df = syn_df[['text','label1','label2']]
    syn_df['label1'] = "Area: " + syn_df['label1']
    syn_df['label2'] = "Recommendation: " + syn_df['label2']
    syn_df.to_csv(f"{target_dir}/PE_iter_{iteration}_augpe.csv",index=False)