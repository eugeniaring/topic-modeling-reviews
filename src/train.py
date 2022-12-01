import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN

import os
import mlflow
from sklearn.ensemble import IsolationForest

from create_data import read_dataset,count_anomalies,create_X_y,read_yaml
from mlflow_log import dagshub_log, get_experiment_id

from ad_if import evaluate_test_if


MODELS_DIR = "models"

def train_bert(docs):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Clustering model: See [2] for more details
    cluster_model = HDBSCAN(min_cluster_size = 15, 
                            metric = 'euclidean', 
                            cluster_selection_method = 'eom', 
                            prediction_data = True)

    # BERTopic model
    topic_model = BERTopic(embedding_model = embedding_model,
                        hdbscan_model = cluster_model)
    # Fit the model on a corpus
    topics, probs = topic_model.fit_transform(docs)
    return topic_model

def visualize_topics(model):
    # Save intertopic distance map as HTML file
    model.visualize_topics().write_html("output/intertopic_dist_map.html")

    # Save topic-terms barcharts as HTML file
    model.visualize_barchart(top_n_topics = 25).write_html("output/barchart.html")

    # Save documents projection as HTML file
    model.visualize_documents(docs).write_html("output/projections.html")

    # Save topics dendrogram as HTML file
    model.visualize_hierarchy().write_html("output/hieararchy.html")    

if __name__ == "__main__":

    params = read_yaml('src/config.yaml')
    mlflow.set_tracking_uri(params['mlflow_url'])
    os.environ['MLFLOW_TRACKING_USERNAME'] = params['MLFLOW_TRACKING_USERNAME']
    os.environ['MLFLOW_TRACKING_PASSWORD'] = params['MLFLOW_TRACKING_PASSWORD']
    df_young = pd.read_csv(params['young_path'],index_col=0)
    docs = df_young['Review Text'].values.tolist() 


    mlflow.tensorflow.autolog()
    with mlflow.start_run():
        topic_model = train_bert(docs)
        visualize_topics(topic_model)





