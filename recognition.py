import cv2
import numpy as np
from datetime import datetime
import os
import logging
import json
from pathlib import Path
from scipy.spatial import distance
from detection import FaceDetectionSystem
from deepface import DeepFace
from sklearn.svm import SVC

class FaceRecognitionSystem:
    def __init__(self):
        
        self.attendance_records = {}
        self.base_threshold = 70  # Base confidence threshold
        self.threshold = self.base_threshold  # Will be adjusted based on lighting
        self.logger = logging.getLogger(__name__)
        self.model_trained = False

        # Recognition classifier
        self.classifier = SVC(kernel='linear', probability=True)
        # self.feature_extracter = DeepFace.represent()
        
        # Loading DNN face recognition model
        self.known_embeddings = []
        self.known_names = []

        # self.classifier.fit(X, y)
        self.face_detector = FaceDetectionSystem()

    def detect_faces(self, frame):
        """Detect and preprocess faces in the frame with improved detection"""
        try:
            # faces, facial_areas, enh_frame = self.face_detector.detect_faces(frame)
            enh_frame = self.face_detector.detect_faces(frame)

            # return faces, facial_areas, enh_frame
            return enh_frame

        except Exception as e:
            print("Detection Error: ", e)
            self.logger.error(f"Error in face detection: {str(e)}")
            return [], [], frame
    
    def train_model(self, faces, labels):
        """Training the face recognition model with enhanced faces and validation"""
        try:
            print("training...")
            if not faces or not labels:
                self.logger.error("No faces or labels provided for training")
                return False
                
            # Preprocessing and preparing training data
            processed_faces = []
            valid_labels = []
            # print("Zip File: ", zip(faces, labels))

            for face_data, label in zip(faces, labels):
                print("Looping Through Faces")
                try:
                    # Check if image is already grayscale
                    # if len(face_data.shape) == 2:  # Already grayscale
                    #     processed_face = face_data
                    # else:  # Convert to grayscale
                    #     processed_face = cv2.cvtColor(face_data, cv2.COLOR_BGR2GRAY)

                                    
                    # if len(face_data.shape) == 2:  # Grayscale
                    #     # print("gray one")
                    #     face_data = cv2.cvtColor(face_data, cv2.COLOR_GRAY2BGR)

                                    
                    # d_faces, facial_areas, enh_frm = self.face_detector.detect_faces(face_data)
                    rep = DeepFace.represent(img_path=f"./student_images/{face_data}", model_name="SFace")

                    # if face_data:
                        # for face in face_data:
                    self.known_embeddings.append(rep[0]['embedding'])
                    self.known_names.append(label)   # student_id
                    
                    processed_faces.append(face_data)
                    valid_labels.append(label)
                    

                except Exception as e:
                    self.logger.warning(f"Skipping invalid face: {str(e)}")
                    continue
            
            
            if not processed_faces:
                self.logger.error("No valid faces found for training")
                return False
            
            print("Processed Faces: ", len(processed_faces))
            print("Valid Labels: ", len(valid_labels))

            
            # X=np.array(self.known_embeddings)
            # y=np.array(self.known_names)
            # self.classifier.fit(X, y)

            self.model_trained = True
            self.logger.info(f"Model trained successfully with {len(processed_faces)} faces")
            return self.known_embeddings, self.known_names
        
        except Exception as e:
            self.model_trained = False
            self.logger.error(f"Error in model training: {str(e)}")
            return False
        
    def get_embeddings(self):

        return self.known_embeddings, self.known_names
    
    def recognize_face(self, frame):
        """Recognizing a face with enhanced preprocessing and adaptive confidence thresholding"""
        try:
            if not self.model_trained:
                print("No trained Model!!")
                self.logger.warning("Model not trained yet. Please train the model first.")
                return None, None
                
            try:
                # Extract Faces
                faces = DeepFace.extract_faces(img_path = frame, detector_backend='retinaface')

                # Compare with known embeddings
                min_dist = float("inf")
                identity = "Unknown"
                
                for face in faces:
                    embedding = DeepFace.represent(face['face'], model_name="SFace")[0]['embedding']
                    pred = self.classifier.predict([embedding])[0]
                    prob = self.classifier.predict_proba([embedding]).max()
                    
                    print("Pred: ", pred, "Prob: ", prob)
                    cv2.putText(frame, f"{pred} ({prob:.2f})", 
                        (face['facial_area']['x'], face['facial_area']['y']-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                
                print("")
                print("Recognizing...")
                # for i, known_face in enumerate(self.known_embeddings):
                #     cv2.imwrite(f"temp_detected{i}.jpg", frame)
                #     cv2.imwrite(f"temp_face{i}.jpg", known_face)

                #     result = DeepFace.verify(f"temp_face{i}.jpg", f"temp_detected{i}.jpg", model_name="SFace")
                #     # print(result)
                #     confidence = result["confidence"]
                #     print("Confidence: ", confidence)

                #     if confidence > 20:
                #         min_dist = confidence
                #         identity = self.known_names[i]                        

                # # Threshold tuning
                # if min_dist > 20:

                #     return identity, min_dist

                # else:

                print("Unknown")
                return None, None


            except cv2.error as e:
                if "not computed yet" in str(e):
                    self.logger.error("Model needs to be trained before recognition")
                    self.model_trained = False
                else:
                    self.logger.error(f"OpenCV error in recognition: {str(e)}")
                return None, None
                
        except Exception as e:
            self.logger.error(f"Error in face recognition: {str(e)}")
            return None, None
    
    def mark_attendance(self, student_id):
        """Mark attendance for a student with validation"""
        try:
            if student_id is None:
                return False
                
            current_date = datetime.now().strftime("%Y-%m-%d")
            current_time = datetime.now().strftime("%H:%M:%S")
            
            if current_date not in self.attendance_records:
                self.attendance_records[current_date] = {}
            
            if student_id not in self.attendance_records[current_date]:
                self.attendance_records[current_date][student_id] = current_time
                self.logger.info(f"Attendance marked for student {student_id} at {current_time}")
                return True
            
            self.logger.info(f"Attendance already marked for student {student_id}")
            return False
        except Exception as e:
            self.logger.error(f"Error in marking attendance: {str(e)}")
            return False
    
    def get_attendance(self, date=None):
        """Get attendance records for a specific date or today"""
        try:
            if date is None:
                date = datetime.now().strftime("%Y-%m-%d")
            
            if date in self.attendance_records:
                return self.attendance_records[date]
            return {}
        except Exception as e:
            self.logger.error(f"Error in getting attendance: {str(e)}")
            return {}
    
    def save_attendance_records(self, filepath='attendance_records.json'):
        """Save attendance records to a JSON file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.attendance_records, f, indent=4)
            self.logger.info(f"Attendance records saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving attendance records: {str(e)}")
            return False
    
    def load_attendance_records(self, filepath='attendance_records.json'):
        """Load attendance records from a JSON file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    self.attendance_records = json.load(f)
                self.logger.info(f"Attendance records loaded from {filepath}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error loading attendance records: {str(e)}")
            return False

# img = cv2.imread("./images/test.png")
# img = cv2.imread("./fromd_1.jpg")
# tester = FaceRecognitionSystem()
# tester.recognize_face(img)