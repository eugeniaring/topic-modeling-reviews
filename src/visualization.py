import pickle
from process_data import read_yaml
from topic_model import load_bert, visualize_topics

if __name__ == "__main__":
    
    params = read_yaml()
    with open('output/docs.pkl', 'rb') as f:
       docs = pickle.load(f)

    topic_model = load_bert(params['model_path'])
    print(topic_model.get_topic_freq().head())
    print('Create visualizations!')
    visualize_topics(topic_model,docs)
    print('End!')