from typing import List

from fastapi import FastAPI, HTTPException

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# from schema import UserGet, PostGet, FeedGet
from schema import PostGet

import os
from catboost import CatBoostClassifier
import pandas as pd


# учетные данные изменены
SQLALCHEMY_DATABASE_URL = "postgresql://dfgdfgdfgdf:dfgfdgfdgdfgfdg@postgres.dfgdgdfgdfgdg:6432/dfgfgfdg"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Загрузка таблицы пользователей
df_users = pd.read_sql("SELECT * FROM user_data", SQLALCHEMY_DATABASE_URL)

# Загрузка таблицы постов с доп. признаками tf-idf/svd
df_posts_from_sql = pd.read_sql('SELECT * FROM i_n_20_idf_posts_lesson_22',
                                con=engine, index_col='index')

# Загрузка файла модели
model_from_file = CatBoostClassifier()
model_from_file.load_model("model_catboost")

app = FastAPI()


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, limit: int = 5) -> List[PostGet]:
    # Данные выбранного user_id:
    user_cols = ['gender', 'age', 'country', 'city', 'exp_group', 'os', 'source']
    cur_user_data = df_users[df_users['user_id'] == id]
    if len(cur_user_data) == 0:
        raise HTTPException(404, "User not found")
    cur_user_data = cur_user_data.drop('user_id', axis=1)

    # Не определяем темы, которые смотрел заданный user_id
    # needed_topic = df_user_id.topic.unique()
    # Берем все темы, т.к. пользователи лайкают много тем, и это ускорит работу приложения
    post_topics = ['business', 'covid', 'entertainment', 'sport', 'politics', 'tech', 'movie']

    # Собираем фрейм для заданного user_id:
    specific_df = pd.DataFrame()

    for cur_topic in post_topics:
        # 200 случайных постов по текущей теме
        temp_df = df_posts_from_sql[df_posts_from_sql['topic'] == cur_topic].sample(n=200)
        # накидываем в общий фрейм для предсказания
        specific_df = pd.concat([specific_df, temp_df], axis=0, ignore_index=True)

    # добавляем во фрейм для предсказания данные выбранного пользователя
    for col in user_cols:
        specific_df[col] = cur_user_data[col].values[0]

    # уберем post_id и text для предсказания
    df_for_predict = specific_df.drop(['post_id', 'text'], axis=1)

    # получаем вероятности для сформированного фрейма
    # и добавляем их в новую колонку
    pred = model_from_file.predict_proba(df_for_predict)
    specific_df['pred'] = pred[:, 1]

    # выделяем топ-5 из полученных вероятностей
    top_pred = specific_df.sort_values(by='pred', ascending=False).head(limit)

    # получаем соответствующие 5 постов
    result = []
    temp_dict = {}
    for cur_id in top_pred['post_id'].values:
        temp_df2 = specific_df[specific_df['post_id'] == cur_id]
        temp_dict = {
                      "id": temp_df2['post_id'].values[0],
                      "text": temp_df2['text'].values[0],
                      "topic": temp_df2['topic'].values[0]
                    }
        result.append(temp_dict)
    if not result:
        raise HTTPException(404, "Post not found")
    else:
        return result
