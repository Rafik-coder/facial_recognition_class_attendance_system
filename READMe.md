

# The Exucutable
The Project is compiled into an EXE Exucutable and stored in the ./builds folder of the project root. Doubleclick on the main.exe to run the project.



# Installations
It is appropriate to first of create a virtual environment.
To create the virtual environment run
`python 
    python -m venv .virt
`
Then run this below to activate it
`python 
    ./.virt/Script/activate
`

The packages needed to run this project is stored in a requirements.txt file, you can install them by running the code below.
`python
 pip install -r requirements.txt
`

# Running the System
After Installations you can run the system using 

`python
    python ./main.py
`

The main.py leads to the lecturer Dashboard. There is a Text Button at the bottom of the Lecturer Login page that leads to the Admin Login Page.

# Project Structure
- builds/
- reports/
- models/
- student_images/
- images/
- main.py
- admin.py
- metrics.py
- database.py
- detection.py
- dialogs.py
- enhance.py
- recognition.py
- operations.py