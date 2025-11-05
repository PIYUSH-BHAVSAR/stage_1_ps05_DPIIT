from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import json, io, zipfile, base64

app = FastAPI()

# ✅ Allow frontend access (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change "*" to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load model once
model = YOLO(r"D:\projects\MULTILINGUAL_OCR\multilingual_doc_understanding\ovr_v2\runs\detect\train3\weights\best.pt")

# ---------- Deskew ----------


def deskew_image(image: Image.Image):
    # Convert PIL → OpenCV (BGR)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert to Grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Hough line transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    if lines is not None:
        angles = []

        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta)

            # Normalize angle to range [-45, 45]
            if angle < 45:
                angles.append(angle)
            elif angle > 135:
                angles.append(angle - 180)

        # Check if angle list is not empty
        if len(angles) > 0:
            median_angle = np.median(angles)

            # Rotate image to deskew
            (h, w) = img_cv.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            
            rotated = cv2.warpAffine(
                img_cv, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )

            # Convert back OpenCV → PIL
            return Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))

    # If no skew is detected → return original
    return image

# ---------- Prediction ----------
def process_image(image: Image.Image):
    deskewed_image = deskew_image(image)
    img_array = np.array(deskewed_image)
    results = model(img_array)
    result = results[0]
    boxes = result.boxes

    annotations = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        class_name = result.names[cls]
        annotations.append(
            {"bbox": [x1, y1, x2, y2], "confidence": conf, "class": class_name}
        )

    annotated_img = img_array.copy()
    for ann in annotations:
        x1, y1, x2, y2 = [int(c) for c in ann["bbox"]]
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated_img,
            f"{ann['class']} {ann['confidence']:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    annotated_pil = Image.fromarray(annotated_img)
    return annotations, annotated_pil


# ---------- Single Image API ----------
@app.post("/predict/single")
async def predict_single(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    annotations, annotated_image = process_image(image)

    img_byte_arr = io.BytesIO()
    annotated_image.save(img_byte_arr, format="PNG")
    encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

    return JSONResponse(content={
        "annotations": annotations,
        "annotated_image_base64": encoded_img
    })
