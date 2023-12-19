import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
video_games_df = pd.read_csv('video_games.csv')
video_games_df['user_review'] = pd.to_numeric(video_games_df['user_review'], errors='coerce')
sns.histplot(video_games_df['user_review'].dropna(), kde=False)
plt.title('Распределение оценок за отзывы пользователей')
plt.xlabel('Оценки')
plt.ylabel('Количество')
plt.show()
video_games_counts = video_games_df['platform'].value_counts()
video_games_counts.plot(kind='bar')
plt.title('Количество видеоигр на каждой платформе')
plt.xlabel('Платформа')
plt.ylabel('Количество')
plt.show()
video_games_df['release_year'] = pd.to_datetime(video_games_df['release_date']).dt.year
sns.countplot(x='release_year', data=video_games_df)
plt.title('Сколько игр выпушено в год')
plt.xlabel('Год')
plt.ylabel('Количество')
plt.xticks(rotation=90)
plt.show()