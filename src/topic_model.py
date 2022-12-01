from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
import mlflow
from dagshub import dagshub_logger

def train_bert(docs,model_path):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Clustering model: See [2] for more details
    cluster_model = HDBSCAN(min_cluster_size = 15, 
                            metric = 'euclidean', 
                            cluster_selection_method = 'eom', 
                            prediction_data = True)

    # BERTopic model
    topic_model = BERTopic(embedding_model = embedding_model,
                        hdbscan_model = cluster_model,language="english")
    # Fit the model on a corpus
    topics, probs = topic_model.fit_transform(docs)
    topic_model.save(model_path)
    return topic_model

def load_bert(model_path):
    topic_model = BERTopic.load(model_path)
    return topic_model

def visualize_topics(model,docs):
    # Save intertopic distance map as HTML file
    model.visualize_topics().write_html("output/intertopic_dist_map.html")
    # mlflow.log_artifact("output/intertopic_dist_map.html")

    # Save topic-terms barcharts as HTML file
    model.visualize_barchart(top_n_topics = 25).write_html("output/barchart.html")
    # mlflow.log_artifact("output/barchart.html")

    # Save documents projection as HTML file
    model.visualize_documents(docs).write_html("output/projections.html")
    # mlflow.log_artifact("output/projections.html")

    # Save topics dendrogram as HTML file
    model.visualize_hierarchy().write_html("output/hieararchy.html") 
    # mlflow.log_artifact("output/hieararchy.html")   