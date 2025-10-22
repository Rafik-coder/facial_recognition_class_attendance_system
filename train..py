
import os
from deepface import DeepFace
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
import cv2
from enhance import ImageEnhancer

data_dir = "datasets"
embeddings = []
labels = []

enhance_image = ImageEnhancer()

i = 1
for student in os.listdir(data_dir):
    student_dir = os.path.join(data_dir, student)
    if not os.path.isdir(student_dir):
        continue
    for img_name in os.listdir(student_dir):
        img_path = os.path.join(student_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = enhance_image.preprocess_image(img)  # preprocess
        try:
            rep = DeepFace.represent(img_path=img_path, model_name="SFace", enforce_detection=False)
            emb = rep[0]["embedding"]
            embeddings.append(emb)
            labels.append(student)
            print(f"Image: {i}")
            i += 1
        except Exception as e:
            print(f"Skipping {img_name}: {e}")


print("Starting...")
# Encode string labels (student IDs)
le = LabelEncoder()
y = le.fit_transform(labels)

# Train SVM classifier
clf = SVC(kernel='linear', probability=True)
clf.fit(embeddings, y)

# Save model and label encoder
joblib.dump(clf, "face_recognition_model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("✅ Model and label encoder saved!")
