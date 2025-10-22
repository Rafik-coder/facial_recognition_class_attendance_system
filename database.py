import sqlite3
import hashlib
import pickle
import cv2
# from scipy.spatial import distance
from detection import FaceDetectionSystem
from deepface import DeepFace

import os, sys

def resource_path(relative_path):
    """Get absolute path to resource (works for dev and for PyInstaller)"""
    try:
        base_path = sys._MEIPASS  # Folder PyInstaller uses for temp extraction
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class DatabaseManager:
    def __init__(self, db_path="attendance.db"):
        self.db_path = resource_path(db_path)
        self.init_database()
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Admin table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS admins (
                id INTEGER PRIMARY KEY,
                admin_id TEXT UNIQUE,
                name TEXT,
                email TEXT,
                password_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Lecturers table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS lecturers (
                id INTEGER PRIMARY KEY,
                lecturer_id TEXT UNIQUE,
                name TEXT,
                email TEXT,
                password_hash TEXT
            )
        ''')
        
        # Courses table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS courses (
                id INTEGER PRIMARY KEY,
                course_code TEXT UNIQUE,
                course_name TEXT,
                lecturer_id TEXT,
                FOREIGN KEY (lecturer_id) REFERENCES lecturers (lecturer_id)
            )
        ''')
        
        # Course enrollments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS course_enrollments (
                id INTEGER PRIMARY KEY,
                course_code TEXT,
                student_id TEXT,
                enrollment_date DATE,
                FOREIGN KEY (course_code) REFERENCES courses (course_code),
                FOREIGN KEY (student_id) REFERENCES students (student_id)
            )
        ''')

        # Students table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY,
                student_id TEXT UNIQUE,
                name TEXT,
                email TEXT,
                image_paths TEXT,
                face_encoding BLOB
            )
        ''')
        
        # Attendance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY,
                student_id TEXT,
                course_code TEXT,
                date DATE,
                time TIME,
                status TEXT,
                FOREIGN KEY (student_id) REFERENCES students (student_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                actual_student_id TEXT,
                predicted_student_id TEXT,
                confidence_score REAL,
                lighting_condition TEXT,
                distance_from_camera REAL,
                face_angle TEXT,
                is_correct_identification BOOLEAN,
                is_accepted BOOLEAN,
                threshold_used REAL
            )
        ''')
        
        # cursor.execute("DROP TABLE IF EXISTS test_results")
        # Add default admin if none exists
        cursor.execute("SELECT COUNT(*) FROM admins")
        if cursor.fetchone()[0] == 0:
            default_password = hashlib.sha256("admin123".encode()).hexdigest()
            cursor.execute('''
                INSERT INTO admins (admin_id, name, email, password_hash)
                VALUES (?, ?, ?, ?)
            ''', ("ADMIN001", "System Administrator", "admin@university.edu", default_password))

        # Add default lecturer if none exists
        cursor.execute("SELECT COUNT(*) FROM lecturers")
        if cursor.fetchone()[0] == 0:
            default_password = hashlib.sha256("admin123".encode()).hexdigest()
            cursor.execute('''
                INSERT INTO lecturers (lecturer_id, name, email, password_hash)
                VALUES (?, ?, ?, ?)
            ''', ("LEC001", "Dr. Admin", "admin@university.edu", default_password))

        # Add default course if none exists
        cursor.execute("SELECT COUNT(*) FROM courses")
        if cursor.fetchone()[0] == 0:
            # Assuming the default lecturer "LEC001" exists
            cursor.execute('''
                INSERT INTO courses (course_code, course_name, lecturer_id)
                VALUES (?, ?, ?)
            ''', ("CS101", "Introduction to Computer Science", "LEC001"))
        
        conn.commit()
        conn.close()
    
    def verify_admin(self, admin_id, password):
        """Verify admin credentials"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        cursor.execute('''
            SELECT * FROM admins 
            WHERE admin_id = ? AND password_hash = ?
        ''', (admin_id, password_hash))
        
        admin = cursor.fetchone()
        conn.close()
        
        return admin is not None

    def add_student(self, student_id, name, email, image_paths):
        """Add a new student to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO students (student_id, name, email, image_paths)
            VALUES (?, ?, ?, ?)
        ''', (student_id, name, email, image_paths))
        
        conn.commit()
        conn.close()
    
    def mark_attendance(self, student_id, course_code):
        """Mark attendance for a student"""
        from datetime import datetime
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now()
        # Normalize to strings to avoid sqlite binding errors for date/time objects
        date_str = now.date().isoformat()          # YYYY-MM-DD
        time_str = now.strftime("%H:%M:%S")       # HH:MM:SS
        try:
            cursor.execute('''
                INSERT INTO attendance (student_id, course_code, date, time, status)
                VALUES (?, ?, ?, ?, ?)
            ''', (student_id, course_code, date_str, time_str, "Present"))
            conn.commit()
        except sqlite3.Error as e:
            # Re-raise with context for upstream handling/logging
            raise Exception(f"Failed to mark attendance: {e}")
        finally:
            conn.close()

    def select(self, cols, table, where=""):
        
        query = f'''
        SELECT {cols} FROM {table} 
        '''
        if len(where) > 0:
            query += f" WHERE {where}"
        
        cmd = self.cursor.execute(query)
        result = cmd.fetchall()

        return result



# db = DatabaseManager()
# students = db.select("student_id, face_encoding", "students")
# face_det = FaceDetectionSystem()

# # # Load DNN face recognition model
# # embedder = cv2.dnn.readNetFromTorch("./face_models/openface_nn4.small2.v1.t7")

# known_embeddings = []
# known_names = []

# for st in students:
#     face_data = pickle.loads(st[1])

#     # Ensure 3-channel image
#     if len(face_data.shape) == 2:  # Grayscale
#         print("gray one")
#         face_data = cv2.cvtColor(face_data, cv2.COLOR_GRAY2BGR)

#     known_embeddings.append(face_data)
#     known_names.append(st[0])
#     # faces, areas, enh_frm = face_det.detect_faces(face_data)
#     # print("Faces: ", faces)
#     print("")
    # print("areas: ", areas)
    # Convert to blob for embedding
    # if faces:
    #     for i, face in enumerate(areas):
    #         x, y, w, h, confid = face
    #         f_roi = face_data[y:y+h, x:x+w]
    #         # cv2.imwrite("from.jpg", f_roi)
    #         # face_blob = cv2.dnn.blobFromImage(
    #         #     f_roi, 
    #         #     scalefactor=1.0/255, 
    #         #     size=(96, 96),   # required size for OpenFace
    #         #     mean=(0, 0, 0), 
    #         #     swapRB=True, 
    #         #     crop=False
    #         # )
    #         # embedder.setInput(face_blob)
    #         # vec = embedder.forward()

    #         cv2.imwrite("")

    #         known_embeddings.append(faces[i])
    #         known_names.append(st[0])   # student_id

# print("Embedings: ", known_embeddings)


# # Load a test image
# img = cv2.imread("./from.jpg")

# if len(img.shape) == 2:  # Grayscale
#     print("gray scae")
#     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# # img = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
# # cv2.imwrite("img.jpg", img)
# # ❌ Wrong: embedder.setInput(img)
# # ✅ Correct: create a blob first

# # Compare with known embeddings
# min_dist = float("inf")
# identity = "Unknown"

# for i, known_face in enumerate(known_embeddings):
#     # cv2.imwrite(f"temp_detected{i}.jpg", face_image)
#     cv2.imwrite(f"temp_face{i}.jpg", known_face)

#     result = DeepFace.verify(f"temp_face{i}.jpg", img, model_name="SFace")
#     print(result)
#     confidence = result["confidence"]
#     if confidence > 0.5:
#         min_dist = confidence
#         identity = known_names[i]

# # Threshold tuning
# if min_dist > 0.5:
#     print(f"Recognized as {identity} (distance={min_dist:.2f})")
# else:
#     print("Unknown")

# print("Done..")

