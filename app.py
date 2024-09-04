from flask import Flask, request, render_template, send_from_directory, url_for
import numpy as np
import cv2
import tensorflow as tf
import os

app = Flask(__name__)

# Load your model
model = tf.keras.models.load_model('gender_classification_model.h5')

TEMP_UPLOAD_FOLDER = 'temp_uploads'
if not os.path.exists(TEMP_UPLOAD_FOLDER):
    os.makedirs(TEMP_UPLOAD_FOLDER)

def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = np.expand_dims(img, axis=0)
    return img

def predict_gender(img):
    img = preprocess_image(img)
    prediction = model.predict(img)
    # Experiment with different thresholds
    threshold = 0.5
    gender = "Male" if prediction < threshold else "Female"
    accuracy = float(prediction) if prediction < threshold else float(1 - prediction)
    return gender, accuracy

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            gender, accuracy = predict_gender(img)
            # Save the uploaded image
            filename = 'uploaded_image.jpg'
            image_path = os.path.join(TEMP_UPLOAD_FOLDER, filename)
            cv2.imwrite(image_path, img)
            return render_template('result.html', gender=gender, accuracy=accuracy, image_url=url_for('uploaded_image', filename=filename))
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_image(filename):
    return send_from_directory(TEMP_UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)