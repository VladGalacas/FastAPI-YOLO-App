import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import cv2
import base64
import io
from PIL import Image
from ultralytics import YOLO

app = FastAPI()
app.mount("static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

device = os.getenv("DEVICE", "cuda")
model = YOLO('best.onnx', task='detect')

@app.post("/process_image/")
async def process_image(request: Request):
    data = await request.json()
    image_base64 = data.get("image")
    if not image_base64:
        return {"error": "No image provided"}

    image_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_data))

    preds = model.predict(image, iou=0.5, conf=0.5, device=device)
    boxes = preds[0].boxes.xyxy.cpu().numpy().astype(int).tolist()
    classes = preds[0].boxes.cls.cpu().numpy().astype(int).tolist()
    conf = preds[0].boxes.conf.cpu().numpy().astype(float).tolist()

    bboxes = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        confidence = conf[i]
        class_id = classes[i]
        bbox_dict = {
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'confidence': confidence,
            'class_id': class_id,
            'class_name': 'Car'
        }
        bboxes.append(bbox_dict)

    result_image = draw_boxes(bboxes, np.array(image))
    buffer = io.BytesIO()
    result_image.save(buffer, format="PNG")
    result_image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {
        "image": result_image_base64,
        "bboxes": bboxes,
    }


def draw_boxes(boxes, image):
    img = image.copy()
    for box in boxes:
        cv2.rectangle(img, (box['x1'], box['y1']), (box['x2'], box['y2']), (255, 0, 0), 2)
        label = f'{box["class_name"]}: {box["confidence"]:.2f}'
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (box['x1'], box['y1'] - text_height - 4), (box['x1'] + text_width, box['y1']),
                      (255, 0, 0), -1)
        cv2.putText(img, label, (box['x1'], box['y1'] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return Image.fromarray(img)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})