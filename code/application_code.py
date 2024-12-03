import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load the saved model
# !!!!!! Change the path accordingly !!!!!!
model_path = 'D:/model/modele_feuilles.h5'
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

# Categories dictionary
categories = [
    "Mango H", "Arjun H", "Alstonia Scholaris H", "Gauva H", "Jamun H", 
    "Jatropha H", "Pongamia Pinnata H", "Basil H", "Pomegranate H", "Lemon H", 
    "Chinar H", "Mango D", "Arjun D", "Alstonia Scholaris D", "Gauva D", 
    "Jamun D", "Jatropha D", "Pongamia Pinnata D", "Basil D", "Pomegranate D", 
    "Lemon D", "Chinar D"
]

def predict_image(image_path):
    try:
        # Load and preprocess the image using PIL
        image = Image.open(image_path).convert('RGB')
        image_resized = image.resize((224, 224))
        image_array = np.array(image_resized) / 255.0
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Make a prediction
        predictions = model.predict(image_array)
        predicted_index = np.argmax(predictions)
        predicted_label = categories[predicted_index]
        confidence = predictions[0][predicted_index] * 100

        print(f"Prediction: {predicted_label} ({confidence:.2f}% confidence)")

    except Exception as e:
        print(f"Error processing the image: {e}")

# Test an image
# !!!!!! Change the path accordingly !!!!!!
test_image_path = 'image.jpg'  # Replace with the path to your test image
if os.path.exists(test_image_path):
    predict_image(test_image_path)
else:
    print(f"The image {test_image_path} does not exist.")
