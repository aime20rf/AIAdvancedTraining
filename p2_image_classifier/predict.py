from PIL import Image
import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

########################
# Function definitions #
########################

# Takes an image as a numpy, resizes to 224x224, and scales to a float value from 0 to 1
def process_image(image):
    tf_image = tf.convert_to_tensor(image)
    tf_image = tf.image.resize(tf_image, (224, 224))
    tf_image = tf.cast(tf_image, tf.float32)
    tf_image /= 255.
    return tf_image.numpy()

# Takes an image path, a loaded model, and an integer top_k, and calculates the top_k 
# most likely classes and associated probabilities
def predict(image_path, model, top_k):
    image = Image.open(image_path)
    processed_image = process_image(np.asarray(image)) # shape (224, 224, 3)
    
    expanded_image = np.expand_dims(processed_image, axis=0) # shape (1, 224, 224, 3)
    prediction = model.predict(expanded_image)[0]
    int_labels = np.flip(np.argsort(prediction)) # gets the integer labels (0 to 101) from highest to lowest priority
    int_labels = int_labels[:top_k] # gets the highest top_k integer labels from highest to top_k-th highest priority
    classes = [str(i+1) for i in int_labels]
    
    probs = np.take(prediction, int_labels)
    return probs, classes

########
# Main #
########

# Parse arguments
parser = argparse.ArgumentParser(
    description='This script uses a pre-trained image classifier to classify flowers.',
)

parser.add_argument('image_path', action="store")
parser.add_argument('saved_model', action="store")
parser.add_argument('--top_k', action="store", nargs='?', metavar='K', default=5, type=int)
parser.add_argument('--category_names', action="store", nargs='?', default='label_map.json')

args = parser.parse_args()

# Load and build model
model = tf.keras.experimental.load_from_saved_model(args.saved_model, custom_objects={'KerasLayer':hub.KerasLayer})
model.build((None, 224, 224, 3))

# Predict top_k probabilities and classes
probs, classes = predict(args.image_path, model, args.top_k)

# Load class names
with open(args.category_names, 'r') as f:
    class_names = json.load(f)
    
# Map classes to names
named_classes = [class_names[key] for key in classes]

# Build result to print
result = {}
for label, probability in zip(named_classes, probs):
    result[label] = '{:.2%}'.format(probability)

# Print results
print("=============================================================================================")
print("The model predicts the following classes with associated probabilities as being most likely: ")
for key, value in result.items():
    print(f"\t{key}: {value}")

