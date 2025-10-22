

import joblib
from deepface import DeepFace
import numpy as np
import cv2
from enhance import ImageEnhancer
from detection import FaceDetectionSystem

enhance_image = ImageEnhancer()
detection = FaceDetectionSystem()

clf = joblib.load("./models/face_recognition_model.pkl")
le = joblib.load("./models/label_encoder.pkl")

def recognize_face(img_path):
    print("Recognizing...")
    # frame = enhance_image.preprocess_image(img_path)
    # faces = detection.detect_faces(img_path)
    # print("Number of Faces detected: ", len(faces))
    # new_overlays = []
    # if(faces):
    #     try:
    #         # Compare with known embeddings
    #         min_dist = float("inf")
    #         identity = "Unknown"
    #         print("")
    #         for face in faces:
    #             x, y, w, h = (
    #                 face["facial_area"]["x"],
    #                 face["facial_area"]["y"],
    #                 face["facial_area"]["w"],
    #                 face["facial_area"]["h"],
    #             )
    #             face_img = img_path[y:y+h, x:x+w]
    #             cv2.imwrite("temp.jpg", face_img)
    #             # img = enhance_image.preprocess_image(face_img)
    #             rep = DeepFace.represent(img_path=face_img, model_name="SFace", enforce_detection=False)
    #             emb = np.array(rep[0]["embedding"]).reshape(1, -1)
    #             pred = clf.predict(emb)[0]
    #             prob = clf.predict_proba(emb).max()
                
    #             print("Pred: ", pred, "Prob: ", prob)
    #             if prob > 0.65:  # you can tune this threshold
    #                 student_index = le.inverse_transform([pred])[0]
    #                 print(f"✅ Recognized as: {student_index} (Confidence: {prob:.2f})")
                    
    #                 label = f"{student_index} ({prob:.2f})"
    #                 color = (0, 255, 0)
    #                 print(f"✅ Recognized as {student_index} with (confidence = {prob})")
    #                 min_dist = prob
    #                 identity = label
                    
    #                 # Side effects: mark attendance and log analytics
    #                 print("")
    #                 # marked = mark_student_attendance(str(student_index))

    #                 # if marked:
    #                 #     try:
    #                 #         print("")
    #                 #         print("Recording Analytics...")
    #                 #         analytics.log_recognition_attempt(
    #                 #             str(student_index), str(student_index), prob,
    #                 #             lighting_condition="normal"
    #                 #         )
    #                 #         print("Analytics Recorded !!!")
    #                 #         print("...")
    #                 #     except Exception:
    #                 #         pass

    #             else:
    #                 label = "Unknown"
    #                 color = (0, 0, 255)
    #                 print("❌ Unknown / Low confidence")
    #                 # try:
    #                 #     # Log an attempt with unknown predicted id
    #                 #     self.analytics.log_recognition_attempt(
    #                 #         actual_id=None,
    #                 #         predicted_id="UNKNOWN",
    #                 #         confidence=prob if prob is not None else 0.0,
    #                 #         lighting_condition="normal"
    #                 #     )
    #                 # except Exception:
    #                 #     pass
    #     except Exception as e:
    #         print(f"Exception {e}")
                

    img = cv2.imread(img_path)
    img = enhance_image.preprocess_image(img)
    rep = DeepFace.represent(img_path=img_path, model_name="SFace", enforce_detection=False)
    emb = np.array(rep[0]["embedding"]).reshape(1, -1)
    pred = clf.predict(emb)[0]
    prob = clf.predict_proba(emb).max()
    
    print("Pred: ", pred, "Prob: ", prob)
    if prob > 0.75:  # you can tune this threshold
        student = le.inverse_transform([pred])[0]
        print(f"✅ Recognized: {student} (Confidence: {prob:.2f})")
    else:
        print("❌ Unknown Face")


# img = './datasets/STU06220715/IMG-20251009-WA0088.jpg'
# img = './datasets/STU06220763/IMG-20251009-WA0159.jpg'
# img = './datasets/STU06220785/IMG-20251009-WA0060.jpg'
img = './images/test.png'
# img = './images/alh.jpg'
# img = './images/31.jpg'
# img = cv2.imread('./datasets/STU062207654/6fcbe9a0f3bd4fa3aa5416658051141c.jpg')
recognize_face(img)

# import sqlite3
# import ast
# import numpy as np
# import cv2
# from deepface import DeepFace
# from sklearn.svm import SVC
# from enhance import ImageEnhancer


# # -------------------------------
# # FACE SYSTEM
# # -------------------------------
# class FaceRecognitionSystem:
#     def __init__(self, detector_backend="retinaface", model_name="Facenet512"):
#         self.detector_backend = detector_backend
#         self.model_name = model_name  # classification-friendly model

#     def detect_faces(self, frame):
#         """Detect faces in a frame and return list of cropped faces"""
#         try:
#             faces = DeepFace.extract_faces(
#                 img_path=frame,
#                 detector_backend=self.detector_backend,
#                 enforce_detection=False
#             )
#             return faces
#         except Exception as e:
#             print("Detection error:", e)
#             return []

#     def embed_face(self, face_input):
#         """
#         Generate an embedding for a given face.
#         Accepts either file path (str) or numpy image (np.ndarray).
#         """
#         try:
#             embedding = DeepFace.represent(
#                 img_path=face_input,
#                 model_name=self.model_name,
#                 detector_backend=self.detector_backend,
#                 enforce_detection=False
#             )[0]["embedding"]
#             return embedding
#         except Exception as e:
#             print("Embedding error:", e)
#             return None


# # -------------------------------
# # DATABASE MANAGER
# # -------------------------------
# class DatabaseManager:
#     def __init__(self, db_path="attendance.db"):
#         self.db_path = db_path


# # -------------------------------
# # MAIN TESTOR CLASS
# # -------------------------------
# class Testor:
#     def __init__(self):
#         self.course_students = []
#         self.face_system = FaceRecognitionSystem()
#         self.db_manager = DatabaseManager()
#         self.image_enhencer = ImageEnhancer()
#         self.model_trained = False
#         self.classifier = SVC(kernel="linear", probability=True)

#         self.load_course_students()
#         self.train_face_model()

#     def load_course_students(self):
#         """Load students enrolled in the current course"""
#         conn = sqlite3.connect(self.db_manager.db_path)
#         cursor = conn.cursor()

#         cursor.execute('''
#             SELECT s.student_id, s.name, s.email, s.image_paths
#             FROM students s
#             JOIN course_enrollments ce ON s.student_id = ce.student_id
#             WHERE ce.course_code = ?
#         ''', ("CS101",))

#         self.course_students = cursor.fetchall()
#         conn.close()

#     def train_face_model(self):
#         """Train the SVM face recognition model with student images"""
#         if not self.course_students:
#             print("⚠️ No students enrolled in this course")
#             return

#         faces = []
#         labels = []

#         for student in self.course_students:
#             face_array = ast.literal_eval(student[-1])  # list of image paths
#             for img_path in face_array:
#                 full_path = f"./student_images/{img_path}"
#                 embedding = self.face_system.embed_face(full_path)
#                 if embedding is not None:
#                     faces.append(embedding)
#                     labels.append(student[0])  # use student_id as label

#         if faces:
#             X = np.array(faces)
#             y = np.array(labels)
#             self.classifier.fit(X, y)
#             self.model_trained = True
#             print(f"✅ Model trained with {len(faces)} face embeddings")
#         else:
#             print("⚠️ No valid face embeddings found for training")

#     def recognize(self, frame):
#         """Recognize faces in a frame"""
#         if not self.model_trained:
#             print("⚠️ Model is not trained")
#             return frame
        
#         frame = self.image_enhencer.preprocess_image(frame)

#         faces = self.face_system.detect_faces(frame)
#         print("Faces detected:", len(faces))

#         for face in faces:
#             face_crop = face["face"]  # numpy array of cropped face
#             embedding = self.face_system.embed_face(face_crop)

#             if embedding is None:
#                 continue

#             pred = self.classifier.predict([embedding])[0]
#             prob = self.classifier.predict_proba([embedding]).max()

#             print("Pred: ", pred, "Prob: ", prob)
#             student_name = next(
#                 (s[1] for s in self.course_students if s[0] == str(pred)), "Unknown"
#             )

#             if prob >= 0.75:  # ✅ stricter threshold
#                 label = f"{student_name} ({prob:.2f})"
#                 color = (0, 255, 0)
#                 print(f"✅ Recognized {student_name} with {prob:.2f}")
#             else:
#                 label = "Unknown"
#                 color = (0, 0, 255)
#                 print("❌ Unknown / Low confidence")

#             # Draw result on frame
#             x, y, w, h = (
#                 face["facial_area"]["x"],
#                 face["facial_area"]["y"],
#                 face["facial_area"]["w"],
#                 face["facial_area"]["h"],
#             )
#             cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#             cv2.putText(frame, label, (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

#         return frame


# # -------------------------------
# # USAGE EXAMPLE
# # -------------------------------
# if __name__ == "__main__":
#     # img = cv2.imread("./images/alh.jpg")  # test image
#     # img = cv2.imread("./images/test.png")
#     img = cv2.imread("./images/31.jpg")
#     testor = Testor()
#     output = testor.recognize(img)

#     cv2.imshow("Recognition", output)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

