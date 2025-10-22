
from cx_Freeze import setup, Executable
import os
import glob

# Get ALL Python files in your project
all_python_files = []
for file in glob.glob("*.py"):
    if file != "setup.py":  # Exclude setup.py itself
        all_python_files.append(file)

# Automatic package detection - cx_Freeze will try to find all imports
build_options = {
    'packages': [],  # Start empty for auto-detection
    'includes': [
        'admin',
        'metrics',
        'database',
        'detection',
        'dialogs',
        'enhance',
        'recognition',
        'operations',
    ],
    'include_files': [
        'admin.py',
        'metrics.py',
        'database.py',
        'detection.py',
        'dialogs.py',
        'enhance.py',
        'recognition.py',
        'operations.py',
        'models/',
        'student_images/',
        'reports/',
        'images/',
    ]  # Add images, data files, etc. here
}

setup(
    name = "STUClassAttendanceSystem",
    version = "1.0",
    description = "This is a Standard Facial Recognition Class Attendance System",
    options = {'build_exe': build_options},
    executables = [Executable("main.py", base="Win32GUI")]  # Hide console on Windows
)