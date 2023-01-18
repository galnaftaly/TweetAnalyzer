import uvicorn
from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import sys
import os 

sys.path.append(os.path.join(os.path.dirname(__file__), 'BGSRD'))
from predict import *

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
     text: str

class TweetList(BaseModel):
     data:List[Tweet]
 
@app.post('/predict')
def predict(tweets: TweetList, dataset: str):
     tweets_list = tweets.data
     df = pd.DataFrame([tweet.dict() for tweet in tweets_list])
     labels, accuracy = get_prediction(df.text.to_list(), dataset)
     accuracy = [eval(format(acc, '.3f')) for acc in accuracy]
     df['label'] = labels
     df['accuracy'] = accuracy
     df['id'] = [i for i in range(1, len(tweets_list) + 1)]
     predictions = df.to_dict(orient="records")
     return predictions
    
