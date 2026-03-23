#!/usr/bin/env python
# coding: utf-8

# In[42]:


from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib as jb

model = jb.load("kmeans_model.pkl")

app = FastAPI(title="Video ID Prediction API")


class YouTubeData(BaseModel):
    Title: str
    Channel_Name: str
    Views: int
    Likes: int
    Comments: int
    Age_In_Days: int
    Engagement_Rate_Percent: float
    Views_Per_Day: float
    Video_URL: str
    Description_Length: int


@app.get("/")
def home():
    return {"message": "API Running"}


@app.post("/predict")
def predict(data:YouTubeData):

    df = pd.DataFrame({
        'Title': [data.Title],
        'Channel_Name': [data.Channel_Name],
        'Views': [data.Views],   
        'Likes': [data.Likes],
        'Comments': [data.Comments],
        'Age_In_Days': [data.Age_In_Days],
        'Engagement_Rate_Percent': [data.Engagement_Rate_Percent],
        'Views_Per_Day': [data.Views_Per_Day],
        'Video_URL': [data.Video_URL],
        'Description_Length': [data.Description_Length]
    })

    prediction = model.predict(df)
    return {"prediction": int(prediction[0])}


# In[ ]:




