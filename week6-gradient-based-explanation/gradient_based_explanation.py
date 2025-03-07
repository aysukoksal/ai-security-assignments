import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

model = VGG16(weights='imagenet')
print("VGG16 model loaded successfully")

#transform your inpur image into a format that VGG16 can process
def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size = (224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis = 0)
    img_array = preprocess_input(img_array)
    return img_array
img_array= load_and_preprocess_image('C:/Users/aysuk/Documents/AISEC/ai-security-assignments/week6-gradient-based-explanation/cat_dog.jpg')
preds = model.predict(img_array)
decoded_preds = decode_predictions(preds, top=3)[0]

#choose the top predicted class for visualization 
target_class_index = np.argmax(preds[0])
print(f"Predicted class: {decoded_preds[0][1]}, class index: {target_class_index}")
img_tensor = tf.convert_to_tensor(img_array)

with tf.GradientTape() as tape:
    tape.watch(img_tensor)
    preds = model(img_tensor)
    target_class_score = preds[0, target_class_index]
gradient = tape.gradient(target_class_score, img_tensor)
gradient = tf.abs(gradient)
saliency_map = tf.reduce_max(gradient, axis = -1)[0]
saliency_map = (saliency_map - tf.reduce_min(saliency_map)) / (tf.reduce_max(saliency_map) - tf.reduce_min(saliency_map))

def show_image_with_saliency(img_path, saliency_map):
    img = image.load_img(img_path, target_size=(224, 224))  # Reload original image
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.imshow(saliency_map, cmap='jet', alpha=0.5)  # Overlay saliency map
    plt.title("Saliency Map")

    plt.show()
show_image_with_saliency('C:/Users/aysuk/Documents/AISEC/ai-security-assignments/week6-gradient-based-explanation/cat_dog.jpg', saliency_map)
