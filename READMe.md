
# Introduction
This is a facial Recognition Class Attendance system that is supposed to give optimal accuracy even under low lightening conditions.

It utilizes Retinex and Histogram Equalization to enhance frames from the video stream before passing in on for detection and recognition. This has proven very accurate by giving up to 94.78% accuracy under deemed environments.

The `main.py` leads to the Lecturer Dashboard, which is where the recognition process takes place. The default login details for it are.
```sh
    LECTURER_ID: LEC001,
    PASSWORD: admin123,
```

The `admin.py` leads to the admin dashboard, which allows managing of data of the system such as, adding, updating and assigning students, lectures and courses.

The defaullt login details to it are
```sh
    ADMIN_ID: ADMIN001
    PASSWORD: admin123
```

# Installations
It is appropriate to first of create a virtual environment.

To create the virtual environment run
```sh
    python -m venv .virt
```

### To Activate
- On Windows, run
```sh 
    ./.virt/Script/activate
```

- On Linux run
```sh
    source .virt/bin/activate
```

# Packages
The packages needed to run this project is stored in a requirements.txt file, you can install them by running the code below.
```sh
 pip install -r requirements.txt
```

# Models
A model was trained using different variations of images of 4 students.

To train a model, create different folders for each student inside the `datasets` folder, labeling each folder with the student index number and store as many images of the student as posible into their various folders. Make sure the images are of different viariations and the difference in number of images for each students to the others do not have vast difference (max 5).

- Traning

The `train.py` is meant for training the model with the students images in the `datasets` folder.

Run it using

```sh
    python ./train.py
```

Uppon successful training, the model will be saved into the `models` folder as `pkl` files, one for faces and another for the coresponding labels.

# Running the System
After Installations and traing of a model, you can run the system using 

for Lecturer Dashboard
```sh
    python ./main.py
```

for Admin Dashboard
```sh
    python ./admin.py
```

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