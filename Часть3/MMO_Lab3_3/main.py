import numpy as np
import pandas as pd
import plotly.express as px

pd.set_option('mode.chained_assignment', None)

df = pd.read_csv("kpop_rankings.csv")
df.rename(columns = {"time": "Week_number"}, inplace = True)
df.head()
songs = pd.DataFrame(df.groupby("song_title")["Week_number"].count()).sort_values(by="Week_number", ascending=False).reset_index()
fig = px.bar(songs.head(10), x = "song_title", y="Week_number")
rank_one = df[df["rank"]==1]
songs = pd.DataFrame(rank_one.groupby("song_title")["Week_number"].count()).sort_values(by="Week_number", ascending=False).reset_index()
fig = px.bar(songs.head(10), x = "song_title", y="Week_number")
fig.update_layout(title = {"text":"Песни, которые были топ 1 наибольшее время", "x":.5, "y":.95},yaxis_title = "number of weeks",plot_bgcolor='white')
fig.update_xaxes(showline=True,linecolor = "black")
fig.update_yaxes(showline=True, linecolor="black")
fig.show()
songs = pd.DataFrame(rank_one.groupby("artist")["Week_number"].count()).sort_values(by="Week_number", ascending=False).reset_index()
fig = px.bar(songs.head(10), x = "artist", y="Week_number")
fig.update_layout(title = {"text":"Исполнители, у которых топ 1 песня дольше всего была в топе", "x":.5, "y":.95},yaxis_title = "number of weeks",plot_bgcolor='white')
fig.update_xaxes(showline=True,linecolor = "black")
fig.update_yaxes(showline=True, linecolor="black")
fig.show()
data_2010 = df[(df["year"]==df["year"].min())&(df["rank"]<=10)]
bar_order = data_2010["artist"].value_counts().reset_index().sort_values(by="count", ascending=False)["artist"].tolist()
fig = px.histogram(data_2010, y= "artist",color = "rank",category_orders={"artist":bar_order})
fig.update_layout(title={"text":"Artists that secured the top 10 position in 2010", "x":0.5,"y":.95},plot_bgcolor='white')
fig.update_xaxes(showline=True,linecolor = "black")
fig.update_yaxes(showline=True, linecolor="black")
fig.show()
data_2023 = df[(df["year"]==df["year"].max())&(df["rank"]<=10)]
bar_order = data_2023["artist"].value_counts().reset_index().sort_values(by="count", ascending=False)["artist"].tolist()
fig = px.histogram(data_2023, y= "artist",color = "rank",category_orders={"artist":bar_order})
fig.update_layout(title={"text":"Artists that secured the top 10 position in 2023", "x":0.5,"y":.95}, plot_bgcolor='white')
fig.update_xaxes(showline=True,linecolor = "black")
fig.update_yaxes(showline=True, linecolor="black")
fig.show()