from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uvicorn
import cv2
import tensorflow as tf
from module.custom_models.custom_unet import *

app = FastAPI()
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
templates = Jinja2Templates(directory="templates")

model = build_unet()
model.load_weights('./models/seg_model_dice.h5')

app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent.absolute() / "static"),
    name="static",
)

#@app.get("/")
#async def root(request: Request):
#    return templates.TemplateResponse(
#        "index.html", {"request": request}
#    )

@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def get_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                await websocket.send_text("some text")
                await websocket.send_bytes(buffer.tobytes())
    except WebSocketDisconnect:
        print("Client disconnected")   
 

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)