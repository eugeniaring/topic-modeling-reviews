from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
import nltk.stem
import joblib

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

def visualize_topics(model,docs,show_plot=False):
    # Save intertopic distance map as HTML file
    fig1 = model.visualize_topics()
    if show_plot == True:
        fig1.show()
    fig1.write_html("output/intertopic_dist_map.html")

    # Save topic-terms barcharts as HTML file
    fig2 = model.visualize_barchart(top_n_topics = 10)
    if show_plot == True:
        fig2.show()
    fig2.write_html("output/barchart.html")

    # Save documents projection as HTML file
    fig3 = model.visualize_documents(docs)
    if show_plot == True:
        fig3.show()
    fig3.write_html("output/projections.html")

    # Save topics dendrogram as HTML file
    fig4 = model.visualize_hierarchy()
    if show_plot == True:
        fig4.show()
    fig4.write_html("output/hierarchy.html")

    fig5 = model.visualize_heatmap(n_clusters=10, width=1000, height=1000)
    if show_plot == True:
        fig5.show()
    fig5.write_html("output/heatmap.html")
