import pandas as pd
import plotly.express as px

data = pd.read_csv('data/processed_data/young_reviews.csv',index_col=0)
#print(data['Age'].describe())
#print((df['Review Text']=='it').sum())

fig_age = px.histogram(data, x = 'Age')
fig_age.write_html("output/age_hist.html")

fig_classname = px.histogram(data, x = data['Class Name'])
fig_classname.write_html("output/classname_hist.html")

data['age_interval'] =  pd.cut(data['Age'],bins=[18,27,31,37],right=False)

fig_classname = px.histogram(data, x = data['Class Name'], color = data['age_interval'])
fig_classname.write_html("output/classname_hist_by_age.html")