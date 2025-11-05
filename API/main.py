from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import io, base64

app = FastAPI()

# ✅ Allow frontend access (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load YOLO model (relative path)
MODEL_PATH = Path(__file__).parent / "model" / "best.pt"
model = YOLO(str(MODEL_PATH))

# ------------------------- DESKEW FUNCTION ------------------------
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
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated, angle


# ------------------------- PREDICTION FUNCTION --------------------
def process_image(image: Image.Image):
    # Convert PIL → CV2 (BGR)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # ✅ Deskew correctly
    deskewed_cv, angle = deskew_image(img_cv)

    # ✅ YOLO inference on deskewed image
    results = model.predict(
        deskewed_cv,
        imgsz=960,
        conf=0.25,
        save=False,
        verbose=False
    )

    result = results[0]
    boxes = result.boxes

    annotations = []
    for box in boxes:
        x_center, y_center, w, h = box.xywh[0].tolist()
        cls_id = int(box.cls[0].item())

        # Convert center → top-left format
        x = float(x_center - w / 2)
        y = float(y_center - h / 2)

        annotations.append({
            "bbox": [x, y, float(w), float(h)],
            "category_id": cls_id + 1  # ✅ 0-index → 1-index
        })

    # ✅ Get annotated deskewed image
    annotated_cv = result.plot()

    # Convert CV2 → PIL
    annotated_pil = Image.fromarray(annotated_cv)

    return annotations, annotated_pil


# ------------------------- API ENDPOINT ---------------------------
@app.post("/predict/single")
async def predict_single(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    annotations, annotated_image = process_image(image)

    img_byte_arr = io.BytesIO()
    annotated_image.save(img_byte_arr, format="PNG")
    encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

    return JSONResponse(content={
        "annotations": annotations,
        "annotated_image_base64": encoded_img
    })
