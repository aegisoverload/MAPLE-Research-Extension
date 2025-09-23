from compute_mauve import *
from pe.embedding.text import SentenceTransformer
import pandas as pd
import re
from collections import Counter
from scipy.spatial.distance import jensenshannon as jsd
from util import set_seed

import argparse
import numpy as np

def compute_category_tvd(series1, series2):
    counts1 = Counter(series1.dropna())
    counts2 = Counter(series2.dropna())
    all_categories = set(counts1.keys()).union(counts2.keys())
    
    dist1 = np.array([counts1.get(cat, 0) for cat in all_categories]) / len(series1)
    dist2 = np.array([counts2.get(cat, 0) for cat in all_categories]) / len(series2)
    
    # Compute Total Variation Distance (TVD)
    tvd = 0.5 * np.sum(np.abs(dist1 - dist2))
    return tvd

def compute_category_jsd(series1, series2):
    counts1 = Counter(series1.dropna())
    counts2 = Counter(series2.dropna())
    all_categories = set(counts1.keys()).union(counts2.keys())
    
    dist1 = np.array([counts1.get(cat, 0) for cat in all_categories]) / len(series1)
    dist2 = np.array([counts2.get(cat, 0) for cat in all_categories]) / len(series2)
    
    return jsd(dist1, dist2)

PMC_METADATA=['medical_specialty', 'clinical_focus',
       'patient_age_group', 'condition_chronicity', 'narrative_structure',
       'primary_intervention_type', 'diagnostic_certainty', 'patient_outcome']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute MAUVE and JSD")
    parser.add_argument("--syn_data_path", help="Path to train data.")
    parser.add_argument("--test_data_path", help="Path to test data.")
    parser.add_argument("--save_dir", help="Path to save results.")
    parser.add_argument("--emb_batch_size", type=int, default=200, help="Embedding batch size")
    args = parser.parse_args()
    print(args)
    
    # Set random seed. Default to be 42.
    set_seed()
    
    # === Load model ===
    embedding = SentenceTransformer(model="sentence-t5-base")
    # # Manually set max_seq_length = 1024
    embedding._model.max_seq_length = 512
    # embedding._model = embedding._model.to('cpu')
    
    df_tst = pd.read_csv(args.test_data_path)
    tst_list = df_tst['text'].tolist()
    p_feats = embedding._model.encode(tst_list, batch_size=args.emb_batch_size)

    results = np.zeros((2+len(BIORXIV_METADATA)+1,)) # MAUVE, word_count JSD, METADATA JSD, AVG JSD
    df_syn = pd.read_csv(args.syn_data_path) 
    syn_list = df_syn['text'].tolist()
    q_feats = embedding._model.encode(syn_list, batch_size=args.emb_batch_size)
    
    # === Compute MAUVE score ===
    results[0] = compute_mauve(p_feats, q_feats)
    
    # === Compute word_count JSD ===
    # Get word count of synthetic texts
    WC_syn = ((df_syn['text'].astype(str).str.split().str.len() / 50).round() * 50).astype(int).to_numpy()
    # Get word count for others
    WC_tst = ((df_tst['text'].astype(str).str.split().str.len() / 50).round() * 50).astype(int).to_numpy()
    df_syn['word_count'] = WC_syn
    df_tst['word_count'] = WC_tst
    word_jsd = compute_category_jsd(df_syn['word_count'], df_tst['word_count'])
    results[1] = np.round(word_jsd,4)
    
    # === Compute other feature TVD alignment ===
    for index, feat_name in enumerate(PMC_METADATA):
        # # For true annotation
        JSD = compute_category_jsd(df_syn['map_'+feat_name], df_tst[feat_name])
        results[2+index] = JSD
        
    avg_jsd = results[iteration,1:-1].mean()
    results[-1] = avg_jsd
    
    print(results)
    np.save(f'{args.save_dir}/MAUVE_JSD.npy',results)

