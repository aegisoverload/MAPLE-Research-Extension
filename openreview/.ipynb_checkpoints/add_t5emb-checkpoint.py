from pe.embedding.text import SentenceTransformer
import pandas as pd

# pre-computing embeding

df_trn = pd.read_csv('./train_dataset_with_metadata_v4.csv')
df_trn = df_trn.rename(columns={'word count':'word_count'})
trn_list = df_trn['text'].tolist()
df_trn['embedding'] = list(embedding._model.encode(trn_list, batch_size=200))
df_trn.to_csv('./train_dataset_with_metadata_v4_t5emb.csv', index=False)
