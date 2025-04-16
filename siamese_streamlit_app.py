import streamlit as st
import cv2
import numpy as np
import face_recognition
import os
import pickle
from PIL import Image
from tensorflow.keras.models import load_model
from siamese_train_model import contrastive_loss

# Load the base model
model = load_model("base_siamese_model.h5", custom_objects={'contrastive_loss': contrastive_loss()})

# Embedding function
def get_embedding(image):
    st.subheader("🔍 Debugging Face Detection")

    face_locations = face_recognition.face_locations(image)
    st.write("📍 Detected face locations:", face_locations)

    if len(face_locations) == 0:
        st.warning("⚠️ No face detected! Try better lighting or centering your face.")
        return None

    top, right, bottom, left = face_locations[0]

    image_with_box = image.copy()
    cv2.rectangle(image_with_box, (left, top), (right, bottom), (0, 255, 0), 2)
    st.image(image_with_box, caption="📸 Face Detected (Bounding Box)", channels="BGR")

    face_image = image[top:bottom, left:right]
    st.write("🧩 Cropped face shape:", face_image.shape)

    face_image = cv2.cvtColor(cv2.resize(face_image, (100, 100)), cv2.COLOR_BGR2GRAY) / 255.0
    embedding = model.predict(np.expand_dims(face_image.reshape(100, 100, 1), axis=0))[0]

    st.write("🧠 Embedding preview (first 5 values):", embedding[:5])
    return embedding

# Load database
def load_database():
    if os.path.exists("embeddings.pkl"):
        with open("embeddings.pkl", "rb") as f:
            return pickle.load(f)
    return {}

# Save database
def save_database(db):
    with open("embeddings.pkl", "wb") as f:
        pickle.dump(db, f)

# Streamlit App
st.title("🧠 Siamese Face Attendance System")

mode = st.sidebar.selectbox("Choose Mode", ["Register Face", "Verify Attendance"])

image_data = st.camera_input("📸 Capture Face")

if image_data is not None:
    image = Image.open(image_data)
    image_np = np.array(image)
    st.image(image_np, caption="🖼️ Captured Face", use_column_width=True)

    embedding = get_embedding(image_np)

    if embedding is None:
        st.warning("⚠️ No valid face embedding generated.")
    else:
        db = load_database()

        if mode == "Register Face":
            name = st.text_input("✏️ Enter Name")

            if name:
                if st.button("💾 Save Face"):
                    db[name] = embedding
                    save_database(db)
                    st.success(f"✅ Face registered for {name}.")
                    st.write(f"🧠 Current registered names: {list(db.keys())}")
            else:
                st.info("📝 Please enter a name before saving.")

        elif mode == "Verify Attendance":
            st.subheader("🔬 Verification Log")
            recognized = False
            closest_name = None
            closest_dist = float('inf')

            for name, stored_embedding in db.items():
                dist = np.linalg.norm(embedding - stored_embedding)
                st.write(f"📏 Distance to {name}: {dist:.4f}")

                if dist < 0.5:
                    if dist < closest_dist:
                        closest_name = name
                        closest_dist = dist
                        recognized = True

            if recognized:
                st.success(f"✅ Attendance Verified: {closest_name} (Distance: {closest_dist:.4f})")
            else:
                st.warning("🚫 Face not recognized. Try re-registering or improving lighting.")

# image1 = cv2.imread("sample1.png", 0)
# image2 = cv2.imread("sample2.png", 0)
# image1 = cv2.resize(image1, (100, 100)) / 255.0
# image2 = cv2.resize(image2, (100, 100)) / 255.0

# embed1 = model.predict(image1.reshape(1,100,100,1))[0]
# embed2 = model.predict(image2.reshape(1,100,100,1))[0]

# distance = np.linalg.norm(embed1 - embed2)
# print("Distance between same person:", distance)
