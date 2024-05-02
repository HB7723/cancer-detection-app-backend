from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

uploads_dir = './uploads'
os.makedirs(uploads_dir, exist_ok=True)

model_path = 'C:/Users/Hridyesh/Documents/App Dev/Shubham/backend/model'
model = tf.saved_model.load(model_path)


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'photo' not in request.files:
        return jsonify({
            'success': False,
            'message': 'No photo part in the request'
        })

    file = request.files['photo']
    if file.filename == '':
        return jsonify({
            'success': False,
            'message': 'No selected file'
        })

    if file and allowed_file(file.filename):
        try:
            image_path = os.path.join(uploads_dir, file.filename)
            image = Image.open(file.stream)
            image.save(os.path.join(uploads_dir, file.filename))
            prediction, error = process_and_predict(image_path)
            print(error)
            if error is not None:
                return jsonify({'success': False, 'message': 'Error in prediction: ' + error})
            print(prediction)
            return jsonify({
                'success': True,
                'message': 'Image uploaded and processed successfully!',
                'predicted_label': int(prediction)
            })
        except IOError as e:
            print(f"Error opening image: {e}")
            return jsonify({
                'success': False,
                'message': 'File is not a valid image'
            })

    return jsonify({
        'success': False,
        'message': 'Invalid file extension'
    })


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}


def process_and_predict(image_path):
    try:
        image = Image.open(image_path)
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = image.astype(np.float32)
        image = np.expand_dims(image, axis=0)

        predictions = model(image, training=False)
        predicted_label = np.argmax(predictions, axis=1)

        return int(predicted_label[0]), None
    except Exception as e:
        return None, str(e)


if __name__ == '__main__':
    app.run(debug=True)
