import os
import mlflow
from dagshub import dagshub_logger

from process_data import read_yaml


if __name__ == "__main__":

    params = read_yaml()
    if params['visualization_over']==False:
        exec(open("src/visualization.py").read())

    l_html = os.listdir('output')  
    l_html.remove('docs.pkl')

    mlflow.set_tracking_uri(params['mlflow_url'])
    os.environ['MLFLOW_TRACKING_USERNAME'] = params['MLFLOW_TRACKING_USERNAME']
    os.environ['MLFLOW_TRACKING_PASSWORD'] = params['MLFLOW_TRACKING_PASSWORD']

    with mlflow.start_run():
        for html_path in l_html:
            mlflow.log_artifact('output/'+html_path)