import pandas as pd

from dagshub.streaming import install_hooks
from process_data import read_yaml

# Install hooks
install_hooks(repo_url='https://dagshub.com/eugenia.anello/topic-modeling-reviews')

# Path of the file to get from the repo
params = read_yaml()
data_path = params['young_path']

# Read the data and print the first 5 rows. 
with open(data_path) as pd_file:
    data = pd.read_csv(pd_file)

print(data.head())