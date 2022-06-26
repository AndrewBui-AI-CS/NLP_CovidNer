import argparse
import base64
import os
import subprocess
from tempfile import NamedTemporaryFile
from typing import List

import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.templating import Jinja2Templates

# import settings

app = FastAPI()
templates = Jinja2Templates(directory='api/templates')
model_selection_options = ['bert-based', 'lstm-crf']
model_dict = {'bert-based': '../model/yolov5.pt',
              'lstm-crf': '../model/pedestrian.yml'}  # set up model cache


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse('home.html', {
        "request": request,
        "model_selection_options": model_selection_options,
    })


@app.post("/")
async def detect_via_web_form(request: Request,
                              textupload: str = Form(...),
                              model_name: str = Form(...),
                              ):

    f = open("data/first_text.txt", "w")
    f.write(textupload)
    f.close()
    # rc = subprocess.call(
    #     ["/home/hoang/Desktop/NLP-Ner/api/call_helper.sh", textupload], shell=True)
    # print("Text up load: ", textupload)
    os.system(f'bash /home/hoang/Desktop/NLP-Ner/api/call_helper.sh \"{textupload}\"')
    first_text = open("data/first_text.txt", "r")
    final_result = open("data/final_result.txt", "r")
    result = [{"first_text": first_text.readlines(
    ), "result": final_result.readlines()}]
    if model_name == 'bert-based':
        return templates.TemplateResponse('show_results.html', {
            'request': request,
            'json_results': result,
        })

    elif model_name == 'lstm-crf':
        return templates.TemplateResponse('show_results.html', {
            'request': request,
            'json_results': result,
        })

if __name__ == '__main__':
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', default=8000)
    opt = parser.parse_args()
    app_str = 'server:app'
    uvicorn.run(app_str, host=opt.host, port=opt.port, reload=True)
