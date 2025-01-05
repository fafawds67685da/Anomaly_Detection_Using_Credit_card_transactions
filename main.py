from fastapi import FastAPI, File, UploadFile
import pandas as p
import numpy as n
import matplotlib.pyplot as m
import io
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

app=FastAPI()
Tree = DecisionTreeClassifier(max_depth=2)
Tree_Accuracy_Score=0

@app.get('/')
def default():
    return {"message": "Backend is working!"}

@app.post("/train/")
async def train(file: UploadFile = File(...)):

    contents = await file.read()

    df = p.read_csv(io.BytesIO(contents))
    x = df.drop(columns='Class', axis=1)
    y = df['Class']

    
    Tree.fit(x,y)
    return {"message": "Training was successful!"}



@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents2 = await file.read()
    data = p.read_csv(io.BytesIO(contents2))
    x2 = data.drop(columns='Class', axis =1)
    y2 = data['Class']

    data['predictions'] = Tree.predict(x2)
    Tree_Accuracy_Score = accuracy_score(y2, data['predictions'])
    output = data.to_csv(index=False).encode('utf-8')

    return StreamingResponse(io.BytesIO(output),
                             media_type='text/csv',
                             headers={"Content-Disposition": "attachment; filename=predictions.csv"})


@app.post("/plot/")
def plot():
    m.figure(figsize=(14, 14))
    tree.plot_tree(Tree,filled=True)
    m.title(f'Decision tree split for the dataset ', color='maroon', fontsize=15)
    buf  = io.BytesIO()
    m.savefig(buf, format='png')
    buf.seek(0)
    m.close()
    return StreamingResponse(buf,
                             media_type="image/png",
                             headers={"Content-Disposition": "attachment; filename=plot.png"})


