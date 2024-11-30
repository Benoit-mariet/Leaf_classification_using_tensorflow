import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Charger le modèle sauvegardé
model_path = 'D:/model/modele_feuilles.h5'
model = tf.keras.models.load_model(model_path)
print("Modèle chargé avec succès.")

# Dictionnaire des catégories
categories = [
    "Mango H", "Arjun H", "Alstonia Scholaris H", "Gauva H", "Jamun H", 
    "Jatropha H", "Pongamia Pinnata H", "Basil H", "Pomegranate H", "Lemon H", 
    "Chinar H", "Mango D", "Arjun D", "Alstonia Scholaris D", "Gauva D", 
    "Jamun D", "Jatropha D", "Pongamia Pinnata D", "Basil D", "Pomegranate D", 
    "Lemon D", "Chinar D"
]

def predict_image(image_path):
    try:
        # Charger et prétraiter l'image avec PIL
        image = Image.open(image_path).convert('RGB')
        image_resized = image.resize((224, 224))
        image_array = np.array(image_resized) / 255.0
        image_array = np.expand_dims(image_array, axis=0)  # Ajouter la dimension batch

        # Faire la prédiction
        predictions = model.predict(image_array)
        predicted_index = np.argmax(predictions)
        predicted_label = categories[predicted_index]
        confidence = predictions[0][predicted_index] * 100

        print(f"Prédiction : {predicted_label} ({confidence:.2f}% de confiance)")

    except Exception as e:
        print(f"Erreur lors du traitement de l'image : {e}")

# Tester une image
test_image_path = 'D:/Plants_2/train/0021_0016.jpg'  # Remplace par le chemin de ton image de test
if os.path.exists(test_image_path):
    predict_image(test_image_path)
else:
    print(f"L'image {test_image_path} n'existe pas.")