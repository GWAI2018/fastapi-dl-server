from typing import Union

from fastapi import FastAPI

from util.helper import prediction_helper

app = FastAPI()


@app.get("/health_check")
def read_root():
    return {"Hello": "World"}

@app.get("/predict/{input}")
async def predict_places(input):
    return_value = prediction_helper(input)
    return {"status": return_value}