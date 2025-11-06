import os
os.environ["OMP_NUM_THREADS"] = "1"       # ✅ reduce CPU threads → lower RAM spikes
os.environ["MKL_NUM_THREADS"] = "1"

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2, io, base64, gc
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ✅ Load YOLO once, CPU only
MODEL_PATH = Path(__file__).parent / "model" / "best.pt"
model = YOLO(str(MODEL_PATH)).to("cpu")
# model.fuse()  # optional: can slightly reduce RAM at inference

def deskew_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_inv = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray_inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) == 0:
        return img, 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    angle = -angle
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle

def process_image(image: Image.Image):
    # ✅ Downscale large images to save RAM (max side = 1280 or 960)
    image.thumbnail((1280, 1280))  # or (960, 960) for even less memory

    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    deskewed_cv, angle = deskew_image(img_cv)

    # ✅ Smaller imgsz to reduce RAM (e.g., 640 instead of 960)
    results = model.predict(
        deskewed_cv,
        imgsz=640,
        conf=0.25,
        save=False,
        verbose=False,
        device="cpu",
    )

    result = results[0]
    boxes = result.boxes

    annotations = []
    for box in boxes:
        x_center, y_center, w, h = box.xywh[0].tolist()
        cls_id = int(box.cls[0].item())
        x = float(x_center - w / 2)
        y = float(y_center - h / 2)
        annotations.append({"bbox": [x, y, float(w), float(h)], "category_id": cls_id + 1})

    annotated_cv = result.plot()

    # ✅ Return JPEG (smaller than PNG) + optional resize to reduce response size
    annotated_cv = cv2.resize(annotated_cv, (annotated_cv.shape[1] // 2, annotated_cv.shape[0] // 2))
    annotated_pil = Image.fromarray(annotated_cv)

    # free intermediate big arrays
    del img_cv, deskewed_cv, result, results, boxes
    gc.collect()

    return annotations, annotated_pil

@app.post("/predict/single")
async def predict_single(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    annotations, annotated_image = process_image(image)

    img_byte_arr = io.BytesIO()
    annotated_image.save(img_byte_arr, format="JPEG", quality=80)  # ✅ smaller than PNG
    encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

    # free more
    del image, annotated_image, img_byte_arr
    gc.collect()

    return JSONResponse(content={"annotations": annotations, "annotated_image_base64": encoded_img})

@app.get("/")
def home():
    return {"status": "OK", "tip": "Use /docs to test"}
