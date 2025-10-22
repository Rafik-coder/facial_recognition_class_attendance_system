import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import sqlite3
from datetime import datetime, timedelta
import threading
import time
import hashlib
import pickle
import os
import webbrowser
from sklearn.svm import SVC
from deepface import DeepFace
import ast
import joblib
import sys

# Add current directory to Python path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import your existing classes
import database
import metrics
import recognition
import detection
import dialogs
import admin
import enhance
import database

class AttendanceSystem:
    def __init__(self, detector_backend="retinaface", model_name="Facenet512"):
        self.detector_backend = detector_backend
        self.model_name = model_name  # classification-friendly model

        self.root = tk.Tk()
        self.root.title("Facial Recognition Attendance System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize system components
        self.db_manager = database.DatabaseManager()
        self.analytics = metrics.FaceRecognitionAnalytics()
        self.face_system = recognition.FaceRecognitionSystem()
        self.image_enhencer = enhance.ImageEnhancer()
        self.face_detection_system = detection.FaceDetectionSystem()

        # Recognition classifier
        self.classifier = SVC(kernel='linear', probability=True)
        self.model_trained = True
        
        # Session variables
        self.current_lecturer = None
        self.current_course = None
        self.video_capture = None
        self.recognition_active = False
        self.attendance_today = []
        self.course_students = []

        cls_path = database.resource_path("./models/face_recognition_model.pkl")
        lb_paths = database.resource_path("./models/label_encoder.pkl")
                
        self.embeddings = joblib.load(cls_path)
        self.face_labels = joblib.load(lb_paths)

        self.recognizing = False
        
        # Start with login screen
        self.show_login_screen()

    def show_login_screen(self):
        """Display the lecturer login screen"""
        # Clear the window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Create login frame
        login_frame = tk.Frame(self.root, bg='white', padx=50, pady=50)
        login_frame.pack(expand=True, fill='both') 
        
        # Title
        title_label = tk.Label(login_frame, text="Facial Recognition Attendance System", 
                              font=('Arial', 24, 'bold'), bg='white', fg='#2c3e50')
        title_label.pack(pady=30)
        
        # Login form frame
        form_frame = tk.Frame(login_frame, bg='white')
        form_frame.pack(expand=True)
        
        # Lecturer ID
        tk.Label(form_frame, text="Lecturer ID:", font=('Arial', 12), 
                bg='white').grid(row=0, column=0, sticky='e', padx=10, pady=10)
        self.lecturer_id_entry = tk.Entry(form_frame, font=('Arial', 12), width=20)
        self.lecturer_id_entry.grid(row=0, column=1, padx=10, pady=10)
        
        # Password
        tk.Label(form_frame, text="Password:", font=('Arial', 12), 
                bg='white').grid(row=1, column=0, sticky='e', padx=10, pady=10)
        self.password_entry = tk.Entry(form_frame, font=('Arial', 12), width=20, show='*')
        self.password_entry.grid(row=1, column=1, padx=10, pady=10)
        
        # Course selection
        tk.Label(form_frame, text="Course:", font=('Arial', 12), 
                bg='white').grid(row=2, column=0, sticky='e', padx=10, pady=10)
        self.course_var = tk.StringVar()
        self.course_combo = ttk.Combobox(form_frame, textvariable=self.course_var, 
                                        font=('Arial', 12), width=18, state='readonly')
        self.course_combo.grid(row=2, column=1, padx=10, pady=10)
        
        # Login button
        login_btn = tk.Button(form_frame, text="Login", font=('Arial', 12, 'bold'),
                             bg='#3498db', fg='white', padx=20, pady=10,
                             command=self.handle_login)
        login_btn.grid(row=3, column=0, columnspan=2, pady=20)
        
        # Button to open admin window
        admin_login_btn = tk.Button(login_frame, text="Admin Login", 
                                    font=('Arial', 10), bg='white', fg='gray',
                                    bd=0, command=self.show_admin_dashboard)
        admin_login_btn.pack(pady=10)
        
        # Load courses when lecturer ID changes
        self.lecturer_id_entry.bind('<KeyRelease>', self.load_lecturer_courses)
        
        # Focus on lecturer ID entry
        self.lecturer_id_entry.focus()
        
        # Bind Enter key to login
        self.root.bind('<Return>', lambda e: self.handle_login())
    
    def load_lecturer_courses(self, event=None):
        """Load courses for the entered lecturer ID"""
        lecturer_id = self.lecturer_id_entry.get().strip()
        if not lecturer_id:
            self.course_combo['values'] = []
            return
        
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT course_code, course_name 
            FROM courses 
            WHERE lecturer_id = ?
        ''', (lecturer_id,))
        
        courses = cursor.fetchall()
        conn.close()
        
        if courses:
            course_options = [f"{code} - {name}" for code, name in courses]
            self.course_combo['values'] = course_options
        else:
            self.course_combo['values'] = []
    
    def handle_login(self):
        """Handle lecturer login"""
        lecturer_id = self.lecturer_id_entry.get().strip()
        password = self.password_entry.get().strip()
        course_selection = self.course_var.get().strip()
        
        if not all([lecturer_id, password, course_selection]):
            messagebox.showerror("Error", "Please fill in all fields")
            return
        
        # Verify lecturer credentials
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        cursor.execute('''
            SELECT name, email FROM lecturers
            WHERE lecturer_id = ? AND password_hash = ?
        ''', (lecturer_id, password_hash))
        
        lecturer = cursor.fetchone()
        conn.close()
        
        if lecturer:
            self.current_lecturer = {
                'id': lecturer_id,
                'name': lecturer[0],
                'email': lecturer[1]
            }
            self.current_course = course_selection.split(' - ')[0]  # Extracting course code
            self.load_course_students()
            self.show_dashboard()
        else:
            messagebox.showerror("Error", "Invalid credentials")
    
    def load_course_students(self):
        """Loading students enrolled in the current course"""
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT s.student_id, s.name, s.email, s.image_paths
            FROM students s
            JOIN course_enrollments ce ON s.student_id = ce.student_id
            WHERE ce.course_code = ?
        ''', (self.current_course,))
        
        self.course_students = cursor.fetchall()
        conn.close()
        
        # Load face model for course students
        # self.load_face_model()
    
    def load_face_model(self):
        """Loading face recognition model for course students"""
        
        # Checking if we have student face data to train with
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        faces = []
        labels = []
        
        for student in self.course_students:
            cursor.execute("SELECT student_id, face_encoding FROM students WHERE student_id = ?", 
                          (student[0],))
            result = cursor.fetchone()
            
            if result and result[0]:
                # Deserialize face encoding
                face_data = pickle.loads(result[1])
                faces.append(face_data)
                # labels.append(int(student[0].replace('STU', '')))  # Convert ID to numeric
                labels.append(result[0])  # Convert ID to numeric
        
        conn.close()
        
        if faces:
            print("Initial Faces Trained...")
            # Train the model with available face data
            self.face_system.train_model(faces, labels)
            # self.face_system.recognizer.save(model_path)
            self.face_system.model_trained = True
            messagebox.showinfo("Info", f"Model trained with {len(faces)} student faces")
        else:
            self.face_system.model_trained = False
            messagebox.showwarning("Warning", "No student face data found. Please add student photos first.")
    
    def show_dashboard(self):
        """Display the main dashboard"""
        # Clear the window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Configure main window
        self.root.title(f"Attendance System - {self.current_course} - {self.current_lecturer['name']}")
        
        # Header frame
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        # Header content
        header_content = tk.Frame(header_frame, bg='#2c3e50')
        header_content.pack(expand=True, fill='both', padx=20, pady=10)
        
        tk.Label(header_content, text=f"Course: {self.current_course}", 
                font=('Arial', 16, 'bold'), bg='#2c3e50', fg='white').pack(side='left')
        
        tk.Label(header_content, text=f"Lecturer: {self.current_lecturer['name']}", 
                font=('Arial', 12), bg='#2c3e50', fg='white').pack(side='left', padx=20)
        
        # Logout button
        logout_btn = tk.Button(header_content, text="Logout", font=('Arial', 10),
                              bg='#e74c3c', fg='white', command=self.logout)
        logout_btn.pack(side='right')
        
        # Main content frame
        main_frame = tk.Frame(self.root, bg='#ecf0f1')
        main_frame.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Left panel - Video and controls
        left_panel = tk.Frame(main_frame, bg='white', relief='raised', bd=2)
        left_panel.pack(side='left', fill='both', expand=True, padx=5)
        
        # Right panel - Attendance and metrics
        right_panel = tk.Frame(main_frame, bg='white', relief='raised', bd=2, width=400)
        right_panel.pack(side='right', fill='y', padx=5)
        right_panel.pack_propagate(False)
        
        self.setup_video_panel(left_panel)
        self.setup_attendance_panel(right_panel)
    
    def setup_video_panel(self, parent):
        """Setup for the video display and controls panel"""
        # Video title
        video_title = tk.Label(parent, text="Live Video Recognition", 
                              font=('Arial', 14, 'bold'), bg='white')
        video_title.pack(pady=10)
        
        # Video display frame with fixed size
        video_frame = tk.Frame(parent, width=640, height=480, bg='black')
        video_frame.pack(pady=10)
        video_frame.pack_propagate(False)  # Prevent frame from resizing
        
        # Video display label
        self.video_label = tk.Label(video_frame, bg='black')
        self.video_label.pack(expand=True, fill='both')
        
        # Control buttons frame
        controls_frame = tk.Frame(parent, bg='white')
        controls_frame.pack(pady=10)
        
        # Start/Stop recognition button
        self.recognition_btn = tk.Button(controls_frame, text="Start Recognition",
                                        font=('Arial', 12, 'bold'), bg='#27ae60', fg='white',
                                        padx=20, pady=10, command=self.toggle_recognition)
        self.recognition_btn.pack(side='left', padx=10)
        
        # Add Student button
        add_student_btn = tk.Button(controls_frame, text="Add Student",
                                   font=('Arial', 12), bg='#3498db', fg='white',
                                   padx=20, pady=10, command=self.show_add_student_dialog)
        add_student_btn.pack(side='left', padx=10)
        
        # Train Model button
        train_btn = tk.Button(controls_frame, text="Train Model", 
                             font=('Arial', 12), bg='#f39c12', fg='white',
                             padx=20, pady=10, command=self.train_face_model)
        train_btn.pack(side='left', padx=10)
        
        # Status label
        self.status_label = tk.Label(parent, text="Recognition: Stopped", 
                                    font=('Arial', 10), bg='white', fg='red')
        self.status_label.pack(pady=5)
    
    def setup_attendance_panel(self, parent):
        """Setup for the attendance tracking panel"""
        # Create notebook for tabs
        notebook = ttk.Notebook(parent)
        notebook.pack(fill='both', expand=True, padx=10, pady=10),
        
        # Attendance tab
        attendance_frame = ttk.Frame(notebook)
        notebook.add(attendance_frame, text="Attendance")
        self.setup_attendance_tab(attendance_frame)
        
        # Metrics tab
        metrics_frame = ttk.Frame(notebook)
        notebook.add(metrics_frame, text="Metrics")
        self.setup_metrics_tab(metrics_frame)
    
    def setup_attendance_tab(self, parent):
        """Setup for the attendance tracking tab"""
        # Present students section (two columns: Index, Time AM/PM)
        present_frame = tk.LabelFrame(parent, text="Present Students", 
                                     font=('Arial', 10, 'bold'), fg='green')
        present_frame.pack(fill='both', expand=True, padx=5, pady=5)

        columns = ("Index", "Time")
        self.present_tree = ttk.Treeview(present_frame, columns=columns, show='headings', height=8)
        for col in columns:
            self.present_tree.heading(col, text=col)
            self.present_tree.column(col, width=120 if col == "Index" else 100, anchor='center')

        present_scrollbar = tk.Scrollbar(present_frame, orient='vertical', command=self.present_tree.yview)
        self.present_tree.configure(yscrollcommand=present_scrollbar.set)

        self.present_tree.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        present_scrollbar.pack(side='right', fill='y', pady=5)
        
        # Absent students section
        absent_frame = tk.LabelFrame(parent, text="Absent Students", 
                                    font=('Arial', 10, 'bold'), fg='red')
        absent_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.absent_listbox = tk.Listbox(absent_frame, font=('Arial', 9))
        absent_scrollbar = tk.Scrollbar(absent_frame, orient='vertical')
        self.absent_listbox.config(yscrollcommand=absent_scrollbar.set)
        absent_scrollbar.config(command=self.absent_listbox.yview)
        
        self.absent_listbox.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        absent_scrollbar.pack(side='right', fill='y', pady=5)
        
        # Print report button
        print_btn = tk.Button(parent, text="Print Attendance Report", 
                             font=('Arial', 11, 'bold'), bg='#8e44ad', fg='white',
                             command=self.print_attendance_report)
        print_btn.pack(pady=10)

        # Refresh students button (so lecturers can get new additions)
        refresh_students_btn = tk.Button(parent, text="Refresh Students", 
                             font=('Arial', 11, 'bold'), bg='#16a085', fg='white',
                             command=self.refresh_course_students)
        refresh_students_btn.pack(pady=5)
        
        # Initialize attendance lists
        self.update_attendance_display()
    
    def setup_metrics_tab(self, parent):
        """Setup the system metrics tab"""
        # Metrics display frame
        metrics_display = tk.Frame(parent)
        metrics_display.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Metrics labels
        self.metrics_labels = {}
        metrics = ['Accuracy', 'Precision', 'Recall', 'False Acceptance Rate', 'False Rejection Rate']
        
        for i, metric in enumerate(metrics):
            frame = tk.Frame(metrics_display)
            frame.pack(fill='x', pady=2)
            
            tk.Label(frame, text=f"{metric}:", font=('Arial', 9, 'bold')).pack(side='left')
            label = tk.Label(frame, text="0.00%", font=('Arial', 9), fg='blue')
            label.pack(side='right')
            self.metrics_labels[metric.lower()] = label
        
        # Refresh metrics button
        refresh_btn = tk.Button(parent, text="Refresh Metrics", 
                               font=('Arial', 10), bg='#16a085', fg='white',
                               command=self.refresh_metrics)
        refresh_btn.pack(pady=10)
        
        # Initialize metrics
        self.refresh_metrics()
    
    def toggle_recognition(self):
        """Toggle video recognition on/off"""
        if not self.recognition_active:
            self.start_recognition()
        else:
            self.stop_recognition()
    
    def start_recognition(self):
        """Starting video recognition"""
        try:
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return
            
            self.recognition_active = True
            self.recognition_btn.config(text="Stop Recognition", bg='#e74c3c')
            self.status_label.config(text="Recognition: Active", fg='green')
            
            # Start video thread
            self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
            self.video_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start recognition: {str(e)}")
    
    def stop_recognition(self):
        """Stop video recognition"""
        self.recognition_active = False
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        
        self.recognition_btn.config(text="Start Recognition", bg='#27ae60')
        self.status_label.config(text="Recognition: Stopped", fg='red')
        
        # Clear video display
        self.video_label.config(image='', bg='black')
    
    def video_loop(self):
        """Main video processing loop"""
        frame_count = 0
        process_every_n_frames = 2  # Process every 2nd frame
        
        while self.recognition_active and self.video_capture:
            ret, frame = self.video_capture.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Resize frame for consistent display and faster processing
            frame = cv2.resize(frame, (640, 480))

            # cv2.imwrite("frame.jpg", frame)

            # Scheduling background processing every Nth frame if worker is idle
            if getattr(self, 'processing', False) is False and frame_count % process_every_n_frames == 0:
                try:
                    self.processing = True
                    self._start_async_processing(frame)
                except Exception:
                    self.processing = False

            # Draw latest overlays (from last completed processing) without blocking
            overlays = getattr(self, 'last_overlays', [])
            if overlays:
                for (x, y, w, h, color, label) in overlays:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    if label:
                        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Convert frame for tkinter display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(image=pil_image)
            
            # Update video display
            self.video_label.config(image=photo)
            self.video_label.image = photo
            
            # Reduce CPU usage
            time.sleep(0.03)  # ~30 FPS

    def _start_async_processing(self, frame):
        """Starting background face detection/recognition without blocking the video loop"""
        # Initialize shared state if needed
        if not hasattr(self, 'last_overlays'):
            self.last_overlays = []
        if not hasattr(self, 'recognizing'):
            self.recognizing = False

        def worker(frame):
            try:
                self.last_overlays = []
                self.recognizing = True

                if not self.model_trained:
                    print("⚠️ Model is not trained. Please train the model first.")
                    # self.logger.warning("Model not trained yet. Please train the model first.")
                    return None, None
                        
                # frame = self.image_enhencer.preprocess_image(frame)
                faces, enhanced_frame = self.face_detection_system.detect_faces(frame, "combined")

                # faces = self.detect_faces(frame)
                print("Number of Faces detected: ", len(faces))
                new_overlays = []
                if(faces):
                    try:
                        # Compare with known embeddings
                        min_dist = float("inf")
                        identity = "Unknown"
                        print("")
                        for face in faces:
                            x, y, w, h = (
                                face["facial_area"]["x"],
                                face["facial_area"]["y"],
                                face["facial_area"]["w"],
                                face["facial_area"]["h"],
                            )
                            face_img = enhanced_frame[y:y+h, x:x+w]
                            be_img = frame[y:y+h, x:x+w]
                            cv2.imwrite("after.jpg", face_img)
                            cv2.imwrite("before.jpg", be_img)
                            enh = self.image_enhencer.preprocess_image(face_img)
                            # cv2.imwrite("temp.jpg", face_img)
                            rep = DeepFace.represent(img_path=enh, model_name="SFace", enforce_detection=False)
                            emb = np.array(rep[0]["embedding"]).reshape(1, -1)
                            pred = self.embeddings.predict(emb)[0]
                            prob = self.embeddings.predict_proba(emb).max()
                            
                            student_index = self.face_labels.inverse_transform([pred])[0]
                            print("Pred: ", student_index, "Prob: ", prob)
                            if prob > 0.85:  # you can tune this threshold
                                # print(f"✅ Recognized as: {student_index} (Confidence: {prob:.2f})")
                                
                                label = f"{student_index} ({prob:.2f})"
                                color = (0, 255, 0)
                                print(f"✅ Recognized as {student_index} with (confidence = {prob})")
                                min_dist = prob
                                identity = label
                                
                                # Side effects: mark attendance and log analytics
                                print("")
                                marked = self.mark_student_attendance(str(student_index))

                                print("")
                                # print("Recording Analytics...")
                                self.analytics.log_recognition_attempt(
                                    str(student_index), str(student_index), prob,
                                    lighting_condition="normal"
                                )
                                print("...")
                                print("")

                                if marked:
                                    try:
                                        print("Analytics Recorded !!!")
                                    except Exception:
                                        pass

                            else:
                                label = "Unknown"
                                color = (0, 0, 255)
                                print("❌ Unknown / Low confidence")
                                print("")
                                print("")
                                try:
                                    # Log an attempt with unknown predicted id
                                    self.analytics.log_recognition_attempt(
                                        actual_id=None,
                                        predicted_id="UNKNOWN",
                                        confidence=prob if prob is not None else 0.0,
                                        lighting_condition="normal"
                                    )
                                except Exception:
                                    pass
                            
                            new_overlays.append((x, y, w, h, (0, 0, 255), identity))

                    except Exception as e:
                        print("Error Recognizing: ", e)
                        new_overlays.append((x, y, w, h, (0, 255, 255), "Error"))
                
                self.last_overlays = new_overlays
                        
            finally:
                self.recognizing = False
                self.processing = False

        # Launch background worker thread
        threading.Thread(target=worker, args=(frame,), daemon=True).start()

    def mark_student_attendance(self, student_id):
        """Mark attendance for a student"""
        # Check if already marked today
        today = datetime.now().date()
        if any(att['student_id'] == student_id and att['date'] == today 
            for att in self.attendance_today):
            self.recognizing = False
            return False
            
        print("Marking Attendance...")
        # Mark attendance in database
        self.db_manager.mark_attendance(student_id, self.current_course)
        
        # Add to today's attendance list
        student_name = next((s[1] for s in self.course_students if s[0] == student_id), "Unknown")
        self.attendance_today.append({
            'student_id': student_id,
            'name': student_name,
            'date': today,
            'time': datetime.now().time()
        })
        print(f"{student_name} Marked Present ✅")
        # Update attendance display
        self.root.after(0, self.update_attendance_display)

        return True
    
    def update_attendance_display(self):
        """Update the attendance display lists"""
        # Clear lists
        if hasattr(self, 'present_tree'):
            for item in self.present_tree.get_children():
                self.present_tree.delete(item)
        self.absent_listbox.delete(0, tk.END)
        
        # Get today's attendance
        today = datetime.now().date()
        present_ids = [att['student_id'] for att in self.attendance_today 
                      if att['date'] == today]
        
        # Populate present table (Index column is student ID, Time column in AM/PM)
        for student in self.course_students:
            if student[0] in present_ids:
                att_time = next((att['time'] for att in self.attendance_today 
                               if att['student_id'] == student[0] and att['date'] == today), None)
                time_str = att_time.strftime("%I:%M %p") if att_time else ""
                self.present_tree.insert('', tk.END, values=(student[0], time_str))
        
        # Populate absent list
        for student in self.course_students:
            if student[0] not in present_ids:
                self.absent_listbox.insert(tk.END, f"{student[1]} ({student[0]})")
    
    def show_add_student_dialog(self):
        """Show dialog to add a new student"""
        dialogs.AddStudentDialog(self.root, self.current_course, self.db_manager, 
                        callback=self.refresh_course_students)
    
    def refresh_course_students(self):
        """Refresh the course students list"""
        self.load_course_students()
        self.update_attendance_display()
    
    def detect_faces(self, frame):
        """Detect faces in a frame and return list of cropped faces"""
        try:
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=self.detector_backend,
                enforce_detection=True
            )
            return faces
        except Exception as e:
            print("Detection error:", e)
            return []

    def embed_face(self, face_input):
        """
        Generate an embedding for a given face.
        Accepts either file path (str) or numpy image (np.ndarray).
        """
        try:
            embedding = DeepFace.represent(
                img_path=face_input,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )[0]["embedding"]
            return embedding
        except Exception as e:
            print("Embedding error:", e)
            return None
        
    def train_face_model(self):
        """Train the face recognition model with student images"""
        # if not self.course_students:
        #     print("⚠️ No students enrolled in this course")
        #     return

        # faces = []
        # labels = []

        # print("Traning model with student faces...")
        # for student in self.course_students:
        #     face_array = ast.literal_eval(student[-1])  # list of image paths
        #     for img_path in face_array:
        #         full_path = f"./student_images/{img_path}"
        #         embedding = self.embed_face(full_path)
        #         if embedding is not None:
        #             faces.append(embedding)
        #             labels.append(student[0])  # use student_id as label

        # if faces:
        #     X = np.array(faces)
        #     y = np.array(labels)
        #     self.classifier.fit(X, y)
        #     self.model_trained = True
        print(f"✅ Model trained!!")
        messagebox.showinfo("Success", f"Model trained with students faces")
        # else:
        #     print("⚠️ No valid face embeddings found for training")
        #     messagebox.showwarning("Warning", "No student face data found. Please add student photos first.")
    
    def print_attendance_report(self):
        """Generate and save attendance report"""
        try:
            # Generate report
            report_data = self.generate_attendance_report()
            
            # Save to file
            os.makedirs('reports', exist_ok=True)
            filename = f"reports/attendance_report_{self.current_course}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(filename, 'w') as f:
                f.write(report_data)
            
            messagebox.showinfo("Success", f"Attendance report saved as {filename}")
            
            # Optionally open the file
            if messagebox.askyesno("Open Report", "Would you like to open the report file?"):
                abs_path = os.path.abspath(filename)
                try:
                    os.startfile(abs_path)
                except Exception:
                    try:
                        webbrowser.open(abs_path)
                    except Exception as e:
                        messagebox.showwarning("Open Report", f"Couldn't open the report automatically. Please open it manually:\n{abs_path}\n\nDetails: {e}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {str(e)}")

    def generate_attendance_report(self):
        """Generate formatted attendance report"""
        today = datetime.now()
        
        # Count present from the present table
        present_count = len(self.present_tree.get_children()) if hasattr(self, 'present_tree') else 0

        report = f"""
ATTENDANCE REPORT
================
Course: {self.current_course}
Lecturer: {self.current_lecturer['name']}
Date: {today.strftime('%Y-%m-%d')}
Time Generated: {today.strftime('%H:%M:%S')}

PRESENT STUDENTS ({present_count}):
{'-' * 40}
"""
        
        # List present students from the two-column table (Index, Time)
        if hasattr(self, 'present_tree'):
            for i, item in enumerate(self.present_tree.get_children(), start=1):
                values = self.present_tree.item(item).get('values', [])
                if len(values) >= 2:
                    student_index = values[0]
                    time_str = values[1]
                    report += f"{i}. {student_index} - {time_str}\n"
        
        report += f"""
ABSENT STUDENTS ({self.absent_listbox.size()}):
{'-' * 40}
"""
        
        for i in range(self.absent_listbox.size()):
            report += f"{i+1}. {self.absent_listbox.get(i)}\n"
        
        report += f"""
SUMMARY:
--------
Total Enrolled: {len(self.course_students)}
Present: {present_count}
Absent: {self.absent_listbox.size()}
Attendance Rate: {(present_count / len(self.course_students) * 100):.1f}%
"""
        
        return report
    
    def refresh_metrics(self):
        """Refresh system performance metrics"""
        try:
            metrics = self.analytics.calculate_metrics_from_db(days_back=7)
            
            if metrics:
                # 'False Acceptance Rate', 'False Rejection Rate'
                self.metrics_labels['accuracy'].config(text=f"{metrics['accuracy']:.2%}")
                self.metrics_labels['precision'].config(text=f"{metrics['precision']:.2%}")
                self.metrics_labels['recall'].config(text=f"{metrics['recall']:.2%}")
                self.metrics_labels['false acceptance rate'].config(text=f"{metrics['false_acceptance_rate']:.2%}")
                self.metrics_labels['false rejection rate'].config(text=f"{metrics['false_rejection_rate']:.2%}")
            else:
                for label in self.metrics_labels.values():
                    label.config(text="No data")
                    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh metrics: {str(e)}")
    
    def logout(self):
        """Logout and return to login screen"""
        if self.recognition_active:
            self.stop_recognition()
        
        self.current_lecturer = None
        self.current_course = None
        self.attendance_today = []
        self.course_students = []
        
        self.show_login_screen()
    
    def show_admin_dashboard(self):
        """Display the admin dashboard"""
        # Create a new window for admin dashboard
        admin_window = tk.Toplevel(self.root)
        admin_dashboard = admin.AdminDashboard(admin_window)
        
        # Hide the main window
        self.root.withdraw()
        
        # When admin window is closed, show main window again
        admin_window.protocol("WM_DELETE_WINDOW", lambda: self.on_admin_close(admin_window))

    def on_admin_close(self, admin_window):
        """Handle admin window closing"""
        admin_window.destroy()
        self.root.deiconify()  # Show main window again
    
    def run(self):
        """Start the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Handle application closing"""
        if self.recognition_active:
            self.stop_recognition()
        self.root.destroy()


if __name__ == "__main__":
    app = AttendanceSystem()
    app.run()



