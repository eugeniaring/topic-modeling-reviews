import pandas as pd
import os
import pickle

import mlflow
from dagshub import dagshub_logger
from dagshub.streaming import install_hooks

from process_data import read_yaml,delete_stopwords
from topic_model import train_bert, load_bert, visualize_topics

if __name__ == "__main__":

    params = read_yaml()
    # Install hooks
    install_hooks(repo_url='https://dagshub.com/eugenia.anello/topic-modeling-reviews')
    
    # remove data from local PC: rm -r data
    print('Load data from remote!')
    with open(params['young_path']) as pd_file:
        df_young = pd.read_csv(pd_file,index_col=0)
    pd_file.close()

    docs = df_young['Review Text'].values.tolist() 
    print(docs[0])
    with open('output/docs.pkl', 'wb') as f:
        pickle.dump(docs, f)
    f.close()

    print('Start training!')
    if params['model_already_trained']==False:
        topic_model = train_bert(docs,params['model_path'])
    else:
        topic_model = load_bert(params['model_path'])
    print('End training!')
    print(topic_model.get_topic_freq().head())



