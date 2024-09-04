### README for Gender Classification Flask App

---

## Overview

This Flask application allows users to upload an image to predict the gender of the person in the image. The prediction is made using a pre-trained deep learning model. The model was trained on the **CelebA dataset**, a large-scale face attributes dataset with over 200,000 celebrity images, each annotated with 40 binary attribute labels.

## Features

- Upload an image for gender classification.
- Display the uploaded image along with the predicted gender.
- Show the accuracy of the prediction.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone (https://github.com/Rohitb0304/Gender-Classification.git)>
cd (https://github.com/Rohitb0304/Gender-Classification.git)>
```

### 2. Install Requirements

Ensure you have Python 3.8 or later installed. Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### 3. Download the Trained Model

Download the pre-trained gender classification model. You can find the model file [here](#). Place the downloaded `gender_classification_model.h5` file in the root directory of the project.

### 4. Download the CelebA Dataset

If you are interested in training or fine-tuning the model, you can download the CelebA dataset from the official source:

- [CelebA Dataset Download Link](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

### 5. Run the Flask App

Start the Flask application by running the following command:

```bash
python app.py
```

### 6. Use the Application

Once the application is running, open your browser and navigate to `http://127.0.0.1:5000/`. Upload an image and see the gender prediction along with the prediction accuracy.

---

## File Structure

```
│
├── app.py                  # The main Flask application
├── gender_classification_model.h5  # Pre-trained model (downloaded separately)
├── requirements.txt        # Python packages required
├── templates/
│   ├── index.html          # HTML for the main page (upload image)
│   └── result.html         # HTML for the result page (display prediction)
└── temp_uploads/           # Temporary folder for storing uploaded images
```

## Model Details

- **Model Architecture:** Custom deep learning model built using TensorFlow.
- **Training Dataset:** CelebA dataset.
- **Input Image Size:** 224x224 pixels.

## Requirements

The application requires the following Python packages:

```plaintext
tensorflow==2.17.0
tensorflow-hub==0.16.1
tensorflow-io-gcs-filesystem==0.37.0
numpy==1.26.0
opencv-python==4.8.1.78
opencv-python-headless==4.10.0.84
matplotlib==3.8.2
matplotlib-inline==0.1.6
```

These versions are verified for compatibility with the current setup.

## Notes

- Ensure that the `gender_classification_model.h5` file is placed in the root directory of the project before running the application.
- The CelebA dataset is large, so be prepared for a significant download if you plan to retrain the model.

---

This README provides all the necessary information to set up and run the Flask-based gender classification application. If you encounter any issues or have questions, feel free to reach out!
