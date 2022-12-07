import pandas as pd
import yaml
from dagshub.streaming import install_hooks

def read_yaml(namefile='src/config.yaml'):
    f = open(namefile,'rb')
    diz = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    return diz

def clean_data(file_path):
    #install_hooks(repo_url='https://dagshub.com/eugenia.anello/topic-modeling-reviews')
    # with open(file_path) as pd_file:
    #     df = pd.read_csv(pd_file,index_col=0)
    # pd_file.close()
    df = pd.read_csv(file_path,index_col=0)
    df.dropna(subset=['Review Text'],inplace=True)
    return df

if __name__ == '__main__':
    params = read_yaml()
    df = clean_data(params['raw_data_path'])
    df_young = df[df.Age<=36]
    df.to_csv(params['all_path'])
    df_young.to_csv(params['young_path'])

