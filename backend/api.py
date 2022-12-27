import uvicorn
from fastapi import FastAPI
from typing import Union,List
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI(title="Tweet Analyzer App")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Tweet(BaseModel):
     id: str 
     tweetText: str

class TweetList(BaseModel):
     data:List[Tweet]  
 
@app.post('/predict')
def predict(tweets: TweetList):
     tweets_list = tweets.data
     df = pd.DataFrame([tweet.dict() for tweet in tweets_list])
     #predictions = model.predict(df)
     df['subject'] = ['Fake News', 'Fake News', 'True News', 'Fake News', 'Fake News']
     df['accuracy'] = [59.1, 68.4, 75.0, 89.7, 93.2]
     df['id'] = [i for i in range(1, len(tweets_list) + 1)]
     predictions = df.to_dict(orient="records")
     print(predictions)
     return predictions
    
