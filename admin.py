import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sqlite3
from datetime import datetime
import hashlib
from database import DatabaseManager
from PIL import Image, ImageTk
import os
import shutil
import cv2
import pickle
from detection import FaceDetectionSystem

class AdminDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Admin Dashboard")
        self.root.geometry("1200x700")

        # Initializing detector
        self.detector = FaceDetectionSystem()
        
        # Initialize database connection
        self.db = DatabaseManager()
        self.conn = sqlite3.connect(self.db.db_path)
        self.cursor = self.conn.cursor()

        self.operating = False
        
        # Create images directory if it doesn't exist
        self.images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'student_images')
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
        
        # Show login screen first
        self.show_login_screen()
    
    def __del__(self):
        """Cleanup database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()
    
    def show_login_screen(self):
        """Display the login screen"""
        # Clear the window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Create login frame
        login_frame = tk.Frame(self.root, bg='white', padx=50, pady=50)
        login_frame.pack(expand=True, fill='both')
        
        # Title
        title_label = tk.Label(login_frame, text="Admin Login", 
                              font=('Arial', 24, 'bold'), bg='white', fg='#2c3e50')
        title_label.pack(pady=30)
        
        # Login form frame
        form_frame = tk.Frame(login_frame, bg='white')
        form_frame.pack(expand=True)
        
        # Username
        tk.Label(form_frame, text="Admin ID:", font=('Arial', 12), 
                bg='white').grid(row=0, column=0, sticky='e', padx=10, pady=10)
        self.username_entry = tk.Entry(form_frame, font=('Arial', 12), width=20)
        self.username_entry.grid(row=0, column=1, padx=10, pady=10)
        
        # Password
        tk.Label(form_frame, text="Password:", font=('Arial', 12), 
                bg='white').grid(row=1, column=0, sticky='e', padx=10, pady=10)
        self.password_entry = tk.Entry(form_frame, font=('Arial', 12), width=20, show='*')
        self.password_entry.grid(row=1, column=1, padx=10, pady=10)
        
        # Login button
        login_btn = tk.Button(form_frame, text="Login", font=('Arial', 12, 'bold'),
                             bg='#3498db', fg='white', padx=20, pady=10,
                             command=self.handle_login)
        login_btn.grid(row=2, column=0, columnspan=2, pady=20)
        
        # Default credentials info
        # info_label = tk.Label(login_frame, text="Default: ADMIN001 / admin123", 
        #                      font=('Arial', 10), bg='white', fg='gray')
        # info_label.pack(pady=10)
        
        # Bind Enter key to login
        self.root.bind('<Return>', lambda e: self.handle_login())
    
    def handle_login(self):
        """Handle admin login"""
        admin_id = self.username_entry.get().strip()
        password = self.password_entry.get().strip()
        
        if not all([admin_id, password]):
            messagebox.showerror("Error", "Please fill in all fields")
            return
        
        # Verify admin credentials using DatabaseManager
        if self.db.verify_admin(admin_id, password):
            self.show_dashboard()
        else:
            messagebox.showerror("Error", "Invalid credentials")
    
    def show_dashboard(self):
        """Display the main dashboard"""
        # Clear the window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Top bar with Logout
        top_bar = tk.Frame(self.root)
        top_bar.pack(fill='x', padx=10, pady=5)
        ttk.Button(top_bar, text="Logout", command=self.show_login_screen).pack(side=tk.RIGHT)

        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=5)
        
        # Create tabs
        self.students_tab = ttk.Frame(self.notebook)
        self.lecturers_tab = ttk.Frame(self.notebook)
        self.courses_tab = ttk.Frame(self.notebook)
        self.enrollments_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.students_tab, text='Students')
        self.notebook.add(self.lecturers_tab, text='Lecturers')
        self.notebook.add(self.courses_tab, text='Courses')
        self.notebook.add(self.enrollments_tab, text='Enrollments')
        
        # Setup each tab
        self.setup_students_tab()
        self.setup_lecturers_tab()
        self.setup_courses_tab()
        self.setup_enrollments_tab()
        
        # Bind tab change event
        self.notebook.bind('<<NotebookTabChanged>>', self.on_tab_changed)
    
    def on_tab_changed(self, event):
        """Handle tab change events"""
        current_tab = self.notebook.select()
        tab_name = self.notebook.tab(current_tab, "text")
        
        if tab_name == "Students":
            self.load_students()
        elif tab_name == "Lecturers":
            self.load_lecturers()
        elif tab_name == "Courses":
            self.load_courses()
        elif tab_name == "Enrollments":
            self.load_students()  # Load students for combobox
            self.load_courses()   # Load courses for combobox
            self.load_enrollments()  # Load enrollments
    
    def setup_students_tab(self):
        # Create frame for student list
        list_frame = ttk.Frame(self.students_tab)
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create Treeview for students
        columns = ('ID', 'Name', 'Email')
        self.students_tree = ttk.Treeview(list_frame, columns=columns, show='headings')
        
        # Set column headings
        for col in columns:
            self.students_tree.heading(col, text=col)
            self.students_tree.column(col, width=100)
        
        self.students_tree.pack(fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.students_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.students_tree.configure(yscrollcommand=scrollbar.set)
        
        # Bind selection event
        self.students_tree.bind('<<TreeviewSelect>>', self.on_student_select)
        
        # Create frame for student form
        form_frame = ttk.LabelFrame(self.students_tab, text="Add/Edit Student")
        form_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)
        
        # Add form fields
        ttk.Label(form_frame, text="Student ID:").pack(pady=5)
        self.student_id_entry = ttk.Entry(form_frame, width=40)
        self.student_id_entry.pack(pady=5)
        
        ttk.Label(form_frame, text="Name:").pack(pady=5)
        self.student_name_entry = ttk.Entry(form_frame, width=40)
        self.student_name_entry.pack(pady=5)
        
        ttk.Label(form_frame, text="Email:").pack(pady=5)
        self.student_email_entry = ttk.Entry(form_frame, width=40)
        self.student_email_entry.pack(pady=5)
        
        # Add image upload section
        image_frame = ttk.LabelFrame(form_frame, text="Student Photo")
        image_frame.pack(pady=10, padx=5, fill=tk.X)
        
        self.image_preview = ttk.Label(image_frame)
        self.image_preview.pack(pady=5)
        
        self.current_image_paths = []
        self.photo_image = None
        
        ttk.Button(image_frame, text="Upload Photo", command=self.upload_student_photos).pack(pady=5)
        
        # Add buttons
        button_frame = ttk.Frame(form_frame, width=40)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Add", command=self.add_student).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Update", command=self.update_student).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Delete", command=self.delete_student).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_student_form).pack(side=tk.LEFT, padx=5)
        
        # Load initial data
        self.load_students()
    
    def setup_lecturers_tab(self):
        # Similar structure to students tab
        list_frame = ttk.Frame(self.lecturers_tab)
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        columns = ('ID', 'Name', 'Email')
        self.lecturers_tree = ttk.Treeview(list_frame, columns=columns, show='headings')
        
        for col in columns:
            self.lecturers_tree.heading(col, text=col)
            self.lecturers_tree.column(col, width=100)
        
        self.lecturers_tree.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.lecturers_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.lecturers_tree.configure(yscrollcommand=scrollbar.set)
        
        # Bind selection event
        self.lecturers_tree.bind('<<TreeviewSelect>>', self.on_lecturer_select)
        
        form_frame = ttk.LabelFrame(self.lecturers_tab, text="Add/Edit Lecturer")
        form_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)
        
        ttk.Label(form_frame, text="Lecturer ID:").pack(pady=5)
        self.lecturer_id_entry = ttk.Entry(form_frame, width=40)
        self.lecturer_id_entry.pack(pady=5)
        
        ttk.Label(form_frame, text="Name:").pack(pady=5)
        self.lecturer_name_entry = ttk.Entry(form_frame, width=40)
        self.lecturer_name_entry.pack(pady=5)
        
        ttk.Label(form_frame, text="Email:").pack(pady=5)
        self.lecturer_email_entry = ttk.Entry(form_frame, width=40)
        self.lecturer_email_entry.pack(pady=5)
        
        ttk.Label(form_frame, text="Password:").pack(pady=5)
        self.lecturer_password_entry = ttk.Entry(form_frame, show="*", width=40)
        self.lecturer_password_entry.pack(pady=5)
        
        button_frame = ttk.Frame(form_frame)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Add", state="", command=self.add_lecturer).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Update", command=self.update_lecturer).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Delete", command=self.delete_lecturer).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_lecturer_form).pack(side=tk.LEFT, padx=5)
        
        self.load_lecturers()
    
    def setup_courses_tab(self):
        list_frame = ttk.Frame(self.courses_tab)
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        columns = ('Code', 'Name', 'Lecturer')
        self.courses_tree = ttk.Treeview(list_frame, columns=columns, show='headings')
        
        for col in columns:
            self.courses_tree.heading(col, text=col)
            self.courses_tree.column(col, width=100)
        
        self.courses_tree.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.courses_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.courses_tree.configure(yscrollcommand=scrollbar.set)
        
        # Bind selection event
        self.courses_tree.bind('<<TreeviewSelect>>', self.on_course_select)
        
        form_frame = ttk.LabelFrame(self.courses_tab, text="Add/Edit Course")
        form_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)
        
        ttk.Label(form_frame, text="Course Code:").pack(pady=5)
        self.course_code_entry = ttk.Entry(form_frame, width=40)
        self.course_code_entry.pack(pady=5)
        
        ttk.Label(form_frame, text="Course Name:").pack(pady=5)
        self.course_name_entry = ttk.Entry(form_frame, width=40)
        self.course_name_entry.pack(pady=5)
        
        ttk.Label(form_frame, text="Lecturer:").pack(pady=5)
        self.course_lecturer_combobox = ttk.Combobox(form_frame, width=40)
        self.course_lecturer_combobox.pack(pady=5)
        
        button_frame = ttk.Frame(form_frame)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Add", command=self.add_course).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Update", command=self.update_course).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Delete", command=self.delete_course).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_course_form).pack(side=tk.LEFT, padx=5)
        
        self.load_courses()
    
    def setup_enrollments_tab(self):
        list_frame = ttk.Frame(self.enrollments_tab)
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        columns = ('Student', 'Course', 'Enrollment Date')
        self.enrollments_tree = ttk.Treeview(list_frame, columns=columns, show='headings')
        
        for col in columns:
            self.enrollments_tree.heading(col, text=col)
            self.enrollments_tree.column(col, width=100)
        
        self.enrollments_tree.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.enrollments_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.enrollments_tree.configure(yscrollcommand=scrollbar.set)
        
        # Bind selection event
        self.enrollments_tree.bind('<<TreeviewSelect>>', self.on_enrollment_select)
        
        form_frame = ttk.LabelFrame(self.enrollments_tab, text="Add Enrollment")
        form_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)
        
        ttk.Label(form_frame, text="Student:").pack(pady=5)
        self.enrollment_student_combobox = ttk.Combobox(form_frame, width=35)
        self.enrollment_student_combobox.pack(pady=5)
        self.enrollment_student_combobox.bind('<<ComboboxSelected>>', self.on_student_selected)
        
        ttk.Label(form_frame, text="Course:").pack(pady=5)
        self.enrollment_course_combobox = ttk.Combobox(form_frame, width=35)
        self.enrollment_course_combobox.pack(pady=5)
        
        button_frame = ttk.Frame(form_frame)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Add", command=self.add_enrollment).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Delete", command=self.delete_enrollment).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_enrollment_form).pack(side=tk.LEFT, padx=5)
        
        # Load data in the correct order
        self.load_students()  # First load students for the combobox
        self.load_courses()   # Then load courses for the combobox
        self.load_enrollments()  # Finally load the enrollments

    def get_enrolled_courses(self, student_id):
        """Get list of course codes that a student is enrolled in"""
        self.cursor.execute("""
            SELECT course_code 
            FROM course_enrollments 
            WHERE student_id = ?
        """, (student_id,))
        return [row[0] for row in self.cursor.fetchall()]

    # Selection handlers
    def on_student_selected(self, event):
        """Handle student selection in enrollment form"""
        student_selection = self.enrollment_student_combobox.get()
        if not student_selection:
            return
            
        # Extract student ID from the selection
        student_id = student_selection.split(' - ')[0]
        
        # Get enrolled courses for this student
        enrolled_courses = self.get_enrolled_courses(student_id)
        
        # Get all courses
        self.cursor.execute("SELECT course_code, course_name FROM courses")
        all_courses = self.cursor.fetchall()
        
        # Filter out enrolled courses
        available_courses = [
            f"{code} - {name}" 
            for code, name in all_courses 
            if code not in enrolled_courses
        ]
        
        # Update course combobox with available courses
        self.enrollment_course_combobox['values'] = available_courses
        self.enrollment_course_combobox.set('')  # Clear current selection

    def on_student_select(self, event):
        selected = self.students_tree.selection()
        if selected:
            values = self.students_tree.item(selected[0])['values']
            # print(values)
            self.student_id_entry.delete(0, tk.END)
            self.student_id_entry.insert(0, values[0])
            self.student_name_entry.delete(0, tk.END)
            self.student_name_entry.insert(0, values[1])
            self.student_email_entry.delete(0, tk.END)
            self.student_email_entry.insert(0, values[2])
            
            # Load student image if exists
            self.cursor.execute("SELECT face_encoding FROM students WHERE student_id = ?", (values[0],))
            result = self.cursor.fetchone()
            if result and result[0] and os.path.exists(result[0]):
                try:
                    image = Image.open(result[0])
                    image = image.resize((150, 150), Image.Resampling.LANCZOS)
                    self.photo_image = ImageTk.PhotoImage(image)
                    self.image_preview.configure(image=self.photo_image)
                    self.current_image_paths.append(result[0])
                except Exception as e:
                    print(f"Failed to load student image: {str(e)}")
    
    def on_lecturer_select(self, event):
        selected = self.lecturers_tree.selection()
        if selected:
            values = self.lecturers_tree.item(selected[0])['values']
            self.lecturer_id_entry.delete(0, tk.END)
            self.lecturer_id_entry.insert(0, values[0])
            self.lecturer_name_entry.delete(0, tk.END)
            self.lecturer_name_entry.insert(0, values[1])
            self.lecturer_email_entry.delete(0, tk.END)
            self.lecturer_email_entry.insert(0, values[2])
            self.lecturer_password_entry.delete(0, tk.END)
    
    def on_course_select(self, event):
        selected = self.courses_tree.selection()
        if selected:
            values = self.courses_tree.item(selected[0])['values']
            self.course_code_entry.delete(0, tk.END)
            self.course_code_entry.insert(0, values[0])
            self.course_name_entry.delete(0, tk.END)
            self.course_name_entry.insert(0, values[1])
            self.course_lecturer_combobox.set(values[2])
    
    def on_enrollment_select(self, event):
        selected = self.enrollments_tree.selection()
        if selected:
            values = self.enrollments_tree.item(selected[0])['values']
            # The values are already in the correct format (ID - Name)
            self.enrollment_student_combobox.set(values[0])
            self.enrollment_course_combobox.set(values[1])
    
    # Database operations
    def load_students(self):
        self.cursor.execute("SELECT student_id, name, email FROM students")
        students = self.cursor.fetchall()

        # Clear and update students tree
        for item in self.students_tree.get_children():
            self.students_tree.delete(item)
        for row in students:
            self.students_tree.insert('', tk.END, values=row)

        # Update enrollment student combobox if it exists
        if hasattr(self, 'enrollment_student_combobox'):
            self.enrollment_student_combobox['values'] = [f"{row[0]} - {row[1]}" for row in students]

    def load_lecturers(self):
        self.cursor.execute("SELECT lecturer_id, name, email FROM lecturers")
        lecturers = self.cursor.fetchall()
        
        # Clear and update lecturers tree
        for item in self.lecturers_tree.get_children():
            self.lecturers_tree.delete(item)
        for row in lecturers:
            self.lecturers_tree.insert('', tk.END, values=row)
        
        # Update lecturer combobox values if it exists
        if hasattr(self, 'course_lecturer_combobox'):
            self.course_lecturer_combobox['values'] = [row[0] for row in lecturers]
    
    def load_courses(self):
        self.cursor.execute("""
            SELECT c.course_code, c.course_name, l.name 
            FROM courses c 
            LEFT JOIN lecturers l ON c.lecturer_id = l.lecturer_id
        """)
        courses = self.cursor.fetchall()
        
        # Clear and update courses tree
        for item in self.courses_tree.get_children():
            self.courses_tree.delete(item)
        for row in courses:
            self.courses_tree.insert('', tk.END, values=row)
        
        # Update course combobox values if it exists
        if hasattr(self, 'enrollment_course_combobox'):
            self.enrollment_course_combobox['values'] = [f"{row[0]} - {row[1]}" for row in courses]
    
    def load_enrollments(self):
        self.cursor.execute("""
            SELECT s.student_id, s.name, c.course_code, c.course_name, e.enrollment_date 
            FROM course_enrollments e
            JOIN students s ON e.student_id = s.student_id
            JOIN courses c ON e.course_code = c.course_code
        """)
        for item in self.enrollments_tree.get_children():
            self.enrollments_tree.delete(item)
        for row in self.cursor.fetchall():
            # Format the display values to match combobox format
            student_display = f"{row[0]} - {row[1]}"
            course_display = f"{row[2]} - {row[3]}"
            self.enrollments_tree.insert('', tk.END, values=(student_display, course_display, row[4]))
    
    # def upload_student_photo(self):
    #     file_path = filedialog.askopenfilename(
    #         title="Select Student Photo",
    #         filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
    #     )
        
    #     if file_path:
    #         try:
    #             # Open and resize image for preview
    #             image = Image.open(file_path)
    #             image = image.resize((150, 150), Image.Resampling.LANCZOS)
    #             self.photo_image = ImageTk.PhotoImage(image)
    #             self.image_preview.configure(image=self.photo_image)
    #             self.current_image_paths.append(str(file_path))

    #         except Exception as e:
    #             messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def upload_student_photos(self):
        # Check if we already have 4 images
        if len(self.current_image_paths) >= 200:
            messagebox.showinfo("Info", "Maximum of 200 photos already uploaded.")
            return
        
        # Calculate how many more images we can select
        remaining_slots = 200 - len(self.current_image_paths)
        
        file_paths = filedialog.askopenfilenames(
            title=f"Select Student Photos ({remaining_slots} remaining)",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        
        if file_paths:
            # Check if selecting these files would exceed the limit
            if len(file_paths) > remaining_slots:
                messagebox.showwarning("Warning", 
                                    f"You can only select {remaining_slots} more photo(s). "
                                    f"Selected {len(file_paths)} files.")
                return
            
            try:
                for file_path in file_paths:
                    # Open and resize image for preview
                    image = Image.open(file_path)
                    image = image.resize((150, 150), Image.Resampling.LANCZOS)
                    photo_image = ImageTk.PhotoImage(image)
                    
                    # Store the image and path
                    self.current_image_paths.append(str(file_path))
                    # Store photo images to prevent garbage collection
                    if not hasattr(self, 'student_images'):
                        self.photo_images = []
                    self.photo_images.append(photo_image)
                
                # Update the preview to show all images
                self.update_image_preview()
                
                # Show status message
                messagebox.showinfo("Success", 
                                f"Added {len(file_paths)} photo(s). "
                                f"Total: {len(self.current_image_paths)}/200")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load images: {str(e)}")
    
    def update_image_preview(self):
        """Update the image preview to show all uploaded images"""
        # Clear previous previews
        if hasattr(self, 'preview_frames'):
            for frame in self.preview_frames:
                frame.destroy()
        
        self.preview_frames = []
        
        # Create a frame to hold all image previews
        if not hasattr(self, 'preview_container'):
            self.preview_container = tk.Frame(self.image_preview.master)
            self.preview_container.pack()
        
        # Display all images
        for i, photo_image in enumerate(getattr(self, 'student_images', [])):
            frame = tk.Frame(self.preview_container)
            frame.grid(row=0, column=i, padx=5)
            
            label = tk.Label(frame, image=photo_image)
            label.pack()
            
            # Add image number
            number_label = tk.Label(frame, text=f"Image {i+1}", font=("Arial", 8))
            number_label.pack()
            
            self.preview_frames.append(frame)


    def add_student(self):
        self.operating = True
        student_id = self.student_id_entry.get()
        name = self.student_name_entry.get()
        email = self.student_email_entry.get()
        
        if not all([student_id, name, email]):
            self.operating = False
            messagebox.showerror("Error", "All fields are required!")
            return
        
        if not self.current_image_paths:
            self.operating = False
            messagebox.showerror("Error", "Please upload a student photo!")
            return
        
        try:
            # Read and process the image
            img_filenames = []
            print("Curent Images: ", self.current_image_paths)
            i = 0
            for image in self.current_image_paths:

                image_d = cv2.imread(image)
                if image_d is None:
                    self.operating = False
                    messagebox.showerror("Error", "Failed to read image file")
                    return
                
                # Copy image to student_images directory for reference
                image_filename = f"{student_id}-{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                image_path = os.path.join(self.images_dir, image_filename)
                shutil.copy2(image, image_path)
                img_filenames.append(image_filename)
                i += 1

            # Add student to database with face encoding
            print("Addding...")
            self.db.add_student(student_id, name, email, str(img_filenames))
            print("Success ✅")
            print('')

            print("Clearing...")
            self.operating = False
            self.load_students()
            self.clear_student_form()
            self.current_image_paths = []
            print("Cleared!!")
            print("")

        except sqlite3.IntegrityError:
            messagebox.showerror("Error", "Student ID already exists!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add student: {str(e)}")
    
    def add_lecturer(self):
        lecturer_id = self.lecturer_id_entry.get()
        name = self.lecturer_name_entry.get()
        email = self.lecturer_email_entry.get()
        password = self.lecturer_password_entry.get()
        
        if not all([lecturer_id, name, email, password]):
            messagebox.showerror("Error", "All fields are required!")
            return
        
        try:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            self.cursor.execute("""
                INSERT INTO lecturers (lecturer_id, name, email, password_hash)
                VALUES (?, ?, ?, ?)
            """, (lecturer_id, name, email, password_hash))
            self.conn.commit()
            self.load_lecturers()
            self.clear_lecturer_form()
        except sqlite3.IntegrityError:
            messagebox.showerror("Error", "Lecturer ID already exists!")
    
    def add_course(self):
        course_code = self.course_code_entry.get()
        course_name = self.course_name_entry.get()
        lecturer_id = self.course_lecturer_combobox.get()
        
        if not all([course_code, course_name, lecturer_id]):
            messagebox.showerror("Error", "All fields are required!")
            return
        
        try:
            self.cursor.execute("""
                INSERT INTO courses (course_code, course_name, lecturer_id)
                VALUES (?, ?, ?)
            """, (course_code, course_name, lecturer_id))
            self.conn.commit()
            self.load_courses()
            self.clear_course_form()
        except sqlite3.IntegrityError:
            messagebox.showerror("Error", "Course code already exists!")
    
    def add_enrollment(self):
        student_selection = self.enrollment_student_combobox.get()
        course_selection = self.enrollment_course_combobox.get()
        
        if not all([student_selection, course_selection]):
            messagebox.showerror("Error", "All fields are required!")
            return
        
        # Extract IDs from the selection strings
        student_id = student_selection.split(' - ')[0]
        course_code = course_selection.split(' - ')[0]
        
        try:
            self.cursor.execute("""
                INSERT INTO course_enrollments (student_id, course_code, enrollment_date)
                VALUES (?, ?, ?)
            """, (student_id, course_code, datetime.now().date()))
            self.conn.commit()
            self.load_enrollments()
            self.clear_enrollment_form()
        except sqlite3.IntegrityError:
            messagebox.showerror("Error", "Enrollment already exists!")
    
    def update_student(self):
        selected = self.students_tree.selection()
        if not selected:
            messagebox.showerror("Error", "Please select a student to update!")
            return
        
        student_id = self.student_id_entry.get()
        name = self.student_name_entry.get()
        email = self.student_email_entry.get()
        
        if not all([student_id, name, email]):
            messagebox.showerror("Error", "All fields are required!")
            return
        
        try:
            self.cursor.execute("""
                UPDATE students 
                SET name = ?, email = ?
                WHERE student_id = ?
            """, (name, email, student_id))
            self.conn.commit()
            self.load_students()
            self.clear_student_form()
        except sqlite3.Error as e:
            messagebox.showerror("Error", str(e))
    
    def update_lecturer(self):
        selected = self.lecturers_tree.selection()
        if not selected:
            messagebox.showerror("Error", "Please select a lecturer to update!")
            return
        
        lecturer_id = self.lecturer_id_entry.get()
        name = self.lecturer_name_entry.get()
        email = self.lecturer_email_entry.get()
        password = self.lecturer_password_entry.get()
        
        if not all([lecturer_id, name, email]):
            messagebox.showerror("Error", "All fields are required!")
            return
        
        try:
            if password:
                password_hash = hashlib.sha256(password.encode()).hexdigest()
                self.cursor.execute("""
                    UPDATE lecturers 
                    SET name = ?, email = ?, password_hash = ?
                    WHERE lecturer_id = ?
                """, (name, email, password_hash, lecturer_id))
            else:
                self.cursor.execute("""
                    UPDATE lecturers 
                    SET name = ?, email = ?
                    WHERE lecturer_id = ?
                """, (name, email, lecturer_id))
            
            self.conn.commit()
            self.load_lecturers()
            self.clear_lecturer_form()
        except sqlite3.Error as e:
            messagebox.showerror("Error", str(e))
    
    def update_course(self):
        selected = self.courses_tree.selection()
        if not selected:
            messagebox.showerror("Error", "Please select a course to update!")
            return
        
        course_code = self.course_code_entry.get()
        course_name = self.course_name_entry.get()
        lecturer_id = self.course_lecturer_combobox.get()
        
        if not all([course_code, course_name, lecturer_id]):
            messagebox.showerror("Error", "All fields are required!")
            return
        
        try:
            self.cursor.execute("""
                UPDATE courses 
                SET course_name = ?, lecturer_id = ?
                WHERE course_code = ?
            """, (course_name, lecturer_id, course_code))
            self.conn.commit()
            self.load_courses()
            self.clear_course_form()
        except sqlite3.Error as e:
            messagebox.showerror("Error", str(e))
    
    def delete_student(self):
        selected = self.students_tree.selection()
        if not selected:
            messagebox.showerror("Error", "Please select a student to delete!")
            return
        
        if messagebox.askyesno("Confirm", "Are you sure you want to delete this student? This will also delete all their course enrollments."):
            student_id = self.students_tree.item(selected[0])['values'][0]
            try:
                # First delete all course enrollments for this student
                self.cursor.execute("DELETE FROM course_enrollments WHERE student_id = ?", (student_id,))
                # Then delete the student
                self.cursor.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
                self.conn.commit()
                self.load_students()
                self.clear_student_form()
            except sqlite3.Error as e:
                messagebox.showerror("Error", str(e))
    
    def delete_lecturer(self):
        selected = self.lecturers_tree.selection()
        if not selected:
            messagebox.showerror("Error", "Please select a lecturer to delete!")
            return
        
        if messagebox.askyesno("Confirm", "Are you sure you want to delete this lecturer?"):
            lecturer_id = self.lecturers_tree.item(selected[0])['values'][0]
            try:
                self.cursor.execute("DELETE FROM lecturers WHERE lecturer_id = ?", (lecturer_id,))
                self.conn.commit()
                self.load_lecturers()
                self.clear_lecturer_form()
            except sqlite3.Error as e:
                messagebox.showerror("Error", str(e))
    
    def delete_course(self):
        selected = self.courses_tree.selection()
        if not selected:
            messagebox.showerror("Error", "Please select a course to delete!")
            return
        
        if messagebox.askyesno("Confirm", "Are you sure you want to delete this course?"):
            course_code = self.courses_tree.item(selected[0])['values'][0]
            try:
                self.cursor.execute("DELETE FROM courses WHERE course_code = ?", (course_code,))
                self.conn.commit()
                self.load_courses()
                self.clear_course_form()
            except sqlite3.Error as e:
                messagebox.showerror("Error", str(e))
    
    def delete_enrollment(self):
        selected = self.enrollments_tree.selection()
        if not selected:
            messagebox.showerror("Error", "Please select an enrollment to delete!")
            return
        
        if messagebox.askyesno("Confirm", "Are you sure you want to delete this enrollment?"):
            # Extract student ID and course code from the selection
            student_display = self.enrollments_tree.item(selected[0])['values'][0]
            course_display = self.enrollments_tree.item(selected[0])['values'][1]
            
            student_id = student_display.split(' - ')[0]
            course_code = course_display.split(' - ')[0]
            
            try:
                self.cursor.execute("""
                    DELETE FROM course_enrollments 
                    WHERE student_id = ? AND course_code = ?
                """, (student_id, course_code))
                self.conn.commit()
                self.load_enrollments()
                self.clear_enrollment_form()
            except sqlite3.Error as e:
                messagebox.showerror("Error", str(e))
    
    def clear_student_form(self):
        self.student_id_entry.delete(0, tk.END)
        self.student_name_entry.delete(0, tk.END)
        self.student_email_entry.delete(0, tk.END)
        self.image_preview.configure(image='')
        self.current_image_paths = []
        self.photo_image = None
    
    def clear_lecturer_form(self):
        self.lecturer_id_entry.delete(0, tk.END)
        self.lecturer_name_entry.delete(0, tk.END)
        self.lecturer_email_entry.delete(0, tk.END)
        self.lecturer_password_entry.delete(0, tk.END)
    
    def clear_course_form(self):
        self.course_code_entry.delete(0, tk.END)
        self.course_name_entry.delete(0, tk.END)
        self.course_lecturer_combobox.set('')
    
    def clear_enrollment_form(self):
        self.enrollment_student_combobox.set('')
        self.enrollment_course_combobox.set('')

if __name__ == "__main__":
    root = tk.Tk()
    app = AdminDashboard(root)
    root.mainloop()