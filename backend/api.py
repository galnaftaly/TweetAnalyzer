import uvicorn
from fastapi import FastAPI
from typing import Union,List
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware




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


class TweetAnalyz(BaseModel):
     id: int
     subject: str
     accuracy: float
     content: str

class TweetList(BaseModel):
     data:List[Tweet]  

class TweetAnalyzList(BaseModel):
     analyzdata: List[TweetAnalyz]  

tweetAnalyz= TweetAnalyzList(analyzdata=[
     { 'id': 1, 'subject':'Fake News', 'accuracy': 59.1, 'content': 'first tweet' },
     { 'id': 2, 'subject':'Fake News', 'accuracy': 68.4, 'content': 'second tweet' },
     { 'id': 3, 'subject':'True News', 'accuracy': 75.0, 'content': 'third tweet' },
     { 'id': 4, 'subject':'Fake News', 'accuracy': 89.7, 'content': 'blalalalg' },
     {
       'id': 5,
       'subject': 'True News',
       'accuracy': 78.4,
       'content': 'fjgkdjlofjfsgdgdhdg',
     } ]  ) 



@app.post('/tweets/')
async def create_tweet(tweets:TweetList):
    #  if len(tweets)>0:
    #      return tweetAnalyz
    return tweetAnalyz
    
