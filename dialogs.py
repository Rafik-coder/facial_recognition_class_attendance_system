import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import pickle
import os

class AddStudentDialog:
    def __init__(self, parent, course_code, db_manager, callback=None):
        self.window = tk.Toplevel(parent)
        self.window.title("Add New Student")
        self.window.geometry("400x500")
        self.window.configure(bg='white')
        
        self.course_code = course_code
        self.db_manager = db_manager
        self.callback = callback
        
        # Create form
        self.create_form()
        
        # Make window modal
        self.window.transient(parent)
        self.window.grab_set()
        
    def create_form(self):
        # Student ID
        tk.Label(self.window, text="Student ID:", bg='white').pack(pady=5)
        self.student_id_entry = tk.Entry(self.window, width=40)
        self.student_id_entry.pack(pady=5)
        
        # Name
        tk.Label(self.window, text="Name:", bg='white').pack(pady=5)
        self.name_entry = tk.Entry(self.window, width=40)
        self.name_entry.pack(pady=5)
        
        # Email
        tk.Label(self.window, text="Email:", bg='white').pack(pady=5)
        self.email_entry = tk.Entry(self.window, width=40)
        self.email_entry.pack(pady=5)
        
        # Photo capture button
        self.capture_btn = tk.Button(self.window, text="Capture Photo", 
                                   command=self.capture_photo)
        self.capture_btn.pack(pady=20)
        
        # Save button
        self.save_btn = tk.Button(self.window, text="Save Student", 
                                command=self.save_student)
        self.save_btn.pack(pady=10)
        
    def capture_photo(self):
        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            return
            
        ret, frame = cap.read()
        if ret:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 1:
                (x, y, w, h) = faces[0]
                face_img = frame[y:y+h, x:x+w]
                self.face_img = cv2.resize(face_img, (200, 200))
                messagebox.showinfo("Success", "Face captured successfully!")
            else:
                messagebox.showerror("Error", "No face detected or multiple faces detected")
                
        cap.release()
        
    def save_student(self):
        student_id = self.student_id_entry.get().strip()
        name = self.name_entry.get().strip()
        email = self.email_entry.get().strip()
        
        if not all([student_id, name, email]):
            messagebox.showerror("Error", "Please fill in all fields")
            return
            
        if not hasattr(self, 'face_img'):
            messagebox.showerror("Error", "Please capture a photo")
            return
            
        try:
            # Convert face image to grayscale
            gray_face = cv2.cvtColor(self.face_img, cv2.COLOR_BGR2GRAY)
            
            # Serialize face encoding
            face_encoding = pickle.dumps(gray_face)
            
            # Save student to database with face encoding
            self.db_manager.add_student(student_id, name, email, face_encoding)
            
            # Enroll student in course
            self.db_manager.enroll_student(student_id, self.course_code)
            
            messagebox.showinfo("Success", "Student added successfully!")
            
            if self.callback:
                self.callback()
                
            self.window.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add student: {str(e)}") 