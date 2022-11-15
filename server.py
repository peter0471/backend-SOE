from typing import Union
import uvicorn

from fastapi import FastAPI, Request, File, UploadFile, BackgroundTasks,Response
from fastapi.middleware.cors import CORSMiddleware

from model import Model
from pydantic import BaseModel
from fastapi import status, HTTPException


app = FastAPI(title="api", debug=True)
fig = Model()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputImage(BaseModel):
    b64:str

class UpdateSet(BaseModel):
    b64:str
    label:str

@app.get("/")
async def read_root():
    return "Root directory"


@app.post("/predict")
async def predict_api(request: Request, image:InputImage):
    res = fig.predict(image.b64)
    if res == -1:
        raise HTTPException(status_code=404)
    else:
        return res

@app.post("/update")
async def update_api(data:UpdateSet):
    print("came here")
    fig.update_dataset(data)
    return ""