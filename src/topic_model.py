from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
import nltk.stem

import mlflow
from dagshub import dagshub_logger

english_stemmer = nltk.stem.SnowballStemmer('english') 
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([english_stemmer.stem(w) for w in analyzer(doc)])

def train_bert(docs,model_path):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Clustering model: See [2] for more details
    cluster_model = HDBSCAN(min_cluster_size = 15, 
                            metric = 'euclidean', 
                            cluster_selection_method = 'eom', 
                            prediction_data = True)
    
    #Explicitly define, use, and adjust the ClassTfidfTransformer with new parameters, 
    #bm25_weighting and reduce_frequent_words, to potentially improve the topic representation
    ctfidf_model = ClassTfidfTransformer(bm25_weighting=True)                         
    #vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2))
    vectorizer_model = StemmedCountVectorizer(analyzer="word",stop_words="english", ngram_range=(1, 2))

    # BERTopic model
    topic_model = BERTopic(embedding_model = embedding_model,
                           hdbscan_model = cluster_model,
                           ctfidf_model=ctfidf_model,
                           vectorizer_model=vectorizer_model,
                           language="english")

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
    model.visualize_barchart(top_n_topics = 10).write_html("output/barchart.html")
    # mlflow.log_artifact("output/barchart.html")

    # Save documents projection as HTML file
    model.visualize_documents(docs).write_html("output/projections.html")
    # mlflow.log_artifact("output/projections.html")

    # Save topics dendrogram as HTML file
    model.visualize_hierarchy().write_html("output/hieararchy.html") 
    # mlflow.log_artifact("output/hieararchy.html")   

    model.visualize_heatmap(n_clusters=10, width=1000, height=1000).write_html("output/heatmap.html") 