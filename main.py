import tensorflow as tf
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel,Field
from contextlib import asynccontextmanager
import joblib

class InputString(BaseModel):
    text:str=Field(min_length=1)

class EmotionResponse(BaseModel):
    Emotion:str
    probability:float
    confidence: str

model=None

@asynccontextmanager
async def lifespan(app:FastAPI):
    global model
    model=tf.keras.models.load_model('Models/bilstm.keras',compile=False)
    print('Model Loaded')

    yield
    print('Shutdown Server')
    model=None

app=FastAPI(title='Emotion Detection API',lifespan=lifespan)

@app.get('/root')
def root():
    return {'Message':'Welcome to Emotion API'}

@app.get('/health')
def health():
    if model:
        return {'Status':'Ready','Models Loaded':True}
    else:
        return {'Status':'Not Ready','Models Loaded':False}
    
@app.post('/predict',response_model=EmotionResponse)
def predict(request:InputString):
    string_input=tf.constant([request.text])
    prediction=model(string_input,training=False,verbose=0).numpy()[0]
    pred_class=int(np.argmax(prediction))
    prob=float(prediction.max())
# 0: sadness 1:joy 2:love 3:anger 4:fear 5:surprise

    if pred_class==0:
        emo='sadness'
    elif pred_class==1:
        emo='joy'
    elif pred_class==2:
        emo='love'
    elif pred_class==3:
        emo='anger'
    elif pred_class==4:
        emo='fear'
    else:
        emo='suprise'

    if prob>0.6:
        confidence='high'
    elif prob>0.3:
        confidence='low confidence'
    else:
        confidence='Neutral/unknown'            


    return EmotionResponse(Emotion=emo,probability=prob,confidence=confidence)
