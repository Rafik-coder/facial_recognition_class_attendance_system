import cv2
import numpy as np
import logging
from skimage import exposure
import imutils
from enhance import ImageEnhancer
from deepface import DeepFace
import time

class FaceDetectionSystem:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialing Immage Enhancer
        self.enhancer = ImageEnhancer()
        
        # Enable timing logs for enhancement to diagnose slowness
        try:
            self.enhancer.debug_timing = True
        except Exception:
            pass
                

    def detect_faces_retinaFace(self, frame, confidence_threshold=0.6):
        try:
            # detect and crop faces
            # detections = DeepFace.extract_faces(img_path=frame, detector_backend="opencv")
            detections = DeepFace.extract_faces(img_path=frame, detector_backend="retinaface")
            # print("Detections: ", detections)
            # looping through all detected faces
            faces = []
            facial_areas = []
            for face in detections:
                print("Faces Detected!!")
                x, y, w, h = (
                    face["facial_area"]["x"],
                    face["facial_area"]["y"],
                    face["facial_area"]["w"],
                    face["facial_area"]["h"],
                )
                face_img = frame[y:y+h, x:x+w]
                confidence = face["confidence"]
                face_img = face["face"]  # numpy array of the cropped face (RGB format)
                # convert RGB → BGR for OpenCV saving
                # face_img_bgr = cv2.cvtColor((face_img * 255).astype("uint8"), cv2.COLOR_RGB2BGR)
                # save each face

                faces.append(face)
                facial_areas.append((x, y, w, h, confidence))

            return faces, facial_areas

        except Exception as e:
            print(f"Error Detecting With RetinaFace: {e}")
            return [], []

    def detect_faces(self, frame, enhancement_method='combined'):
        try:
            """
            Draw rectangles around detected faces
            """
            # Enhance Frame (timed inside enhancer when debug_timing=True)
            enhanced_frame = self.enhancer.preprocess_image(frame, enhancement_method)

            # t0 = time.time()
            faces, facial_areas = self.detect_faces_retinaFace(enhanced_frame)
            # t1 = time.time()
            # if hasattr(self.enhancer, 'debug_timing') and self.enhancer.debug_timing:
            #     self.logger.info(f"RetinaFace extraction took {(t1 - t0)*1000:.1f} ms; faces={len(faces)}")
            
            # return faces, facial_areas, enhanced_frame
            return faces, enhanced_frame

        except Exception as e:
            print(f"Error detecting Face: {e}")
            return [], [], frame
    
# img = cv2.imread("./images/pic2.jpg")
# tester = FaceDetectionSystem()
# # retinex, combined
# tester.detect_faces(img, "retinex") 
# if faces:
#     face = tester.detect_faces(img.copy(), faces)
#     cv2.imwrite("./face1.jpg", face)
#     print("Faces detected and saved at ./images/faces_detected.jpg")