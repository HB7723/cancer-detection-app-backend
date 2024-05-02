import tensorflow as tf
import numpy as np
from PIL import Image

# Path to the directory containing the saved model and the image
model_path = 'C:/Users/Hridyesh/Documents/App Dev/Shubham/backend/model'
image_path = 'C:/Users/Hridyesh/Documents/App Dev/Shubham/backend/uploads/upload.jpg'

# Load the saved model
model = tf.saved_model.load(model_path)

# Load and preprocess the image
image = Image.open(image_path)
# Resize the image to match the model's expected input
image = image.resize((224, 224))
image = np.array(image) / 255.0   # Scale pixel values to [0, 1]
image = image.astype(np.float32)  # Convert image to float32
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Make sure to set the training flag to False if your model uses it
# Check if the model expects a specific function call
try:
    # Adjust based on the expected input in the error
    predictions = model(image, training=False)
except Exception as e:
    print("Error during model prediction:", e)

predicted_label = np.argmax(predictions, axis=1)
print("Predicted Label:", predicted_label)
