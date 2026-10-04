import cv2
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------- CONFIG ----------------
MODEL_PATH = "mask_detector.h5"
IMG_SIZE = 224
MASK_INDEX = 0  # set to 1 if your labels come out swapped
# ----------------------------------------

st.set_page_config(page_title="Face Mask Detection", page_icon="😷", layout="centered")


@st.cache_resource
def load_assets():
    model = load_model(MODEL_PATH)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    return model, cascade


model, face_cascade = load_assets()


def classify(face_rgb):
    face = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE)).astype("float32")
    face = preprocess_input(face)
    preds = model.predict(np.expand_dims(face, 0), verbose=0)[0]
    if preds.shape[0] == 1:
        p_mask = float(preds[0]) if MASK_INDEX == 0 else 1 - float(preds[0])
    else:
        p_mask = float(preds[MASK_INDEX])
    return ("Mask", p_mask) if p_mask >= 0.5 else ("No Mask", 1 - p_mask)


def analyze(pil_img):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    annotated = img.copy()
    results = []

    if len(faces) == 0:
        label, conf = classify(img)
        results.append((label, conf))
    else:
        for (x, y, w, h) in faces:
            label, conf = classify(img[y:y + h, x:x + w])
            results.append((label, conf))
            color = (0, 200, 0) if label == "Mask" else (220, 0, 0)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
            cv2.putText(annotated, f"{label} {conf*100:.0f}%", (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return annotated, results, len(faces)


def show_result(pil_img):
    annotated, results, n_faces = analyze(pil_img)
    st.image(annotated, use_container_width=True)

    if n_faces == 0:
        st.info("No face detected, so the whole image was analyzed.")

    for i, (label, conf) in enumerate(results, 1):
        title = f"Person {i}: " if len(results) > 1 else ""
        if label == "Mask":
            st.success(f"{title}😷 Mask Detected ({conf*100:.1f}% confidence)")
        else:
            st.error(f"{title}🚫 No Mask Detected ({conf*100:.1f}% confidence)")


st.title("😷 Face Mask Detection")
st.caption("Upload an image or use your webcam to check whether a person is wearing a mask.")

tab_upload, tab_webcam = st.tabs(["📁 Upload", "📷 Webcam"])

with tab_upload:
    file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if file:
        show_result(Image.open(file))

with tab_webcam:
    photo = st.camera_input("Take a photo")
    if photo:
        show_result(Image.open(photo))
