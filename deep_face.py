# Keras model classes
from keras.models import Model, Sequential, model_from_json
from keras.layers import Input, Convolution2D, MaxPooling2D, ZeroPadding2D, Flatten, Dense, Dropout, Activation

# Keras iamge processing
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

# Image manipulation libraries
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import cv2
import os
import time

blue_color = (52, 148, 230)
pink_color = (236, 110, 173)

# VGGFace Model
model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

# Add pre-train weights for model
model.load_weights("vgg-weights.h5")
vgg_descriptor = Model(
    inputs=model.layers[0].input, outputs=model.layers[-2].output)


def process_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def find_cos_sim(source_representation, test_representation):
    x = np.matmul(np.transpose(source_representation), test_representation)
    y = np.sum(np.multiply(source_representation, source_representation))
    z = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (x / (np.sqrt(y) * np.sqrt(z)))


encode_start = time.time()
faces_to_match_against = "PATH_TO_FACE_IMAGES_FOLDERS"
faces = dict()

directories = os.listdir(faces_to_match_against)
for directory in directories:
    person_dir = faces_to_match_against + "/" + directory
    person_images = os.listdir(person_dir)
    for n_image in person_images:
        if n_image.startswith("."):
            continue
        file_name = person_dir + "/" + n_image
        face, extension = n_image.split(".")
        faces[face] = vgg_descriptor.predict(process_image(file_name))[0, :]
encode_end = time.time()
print(f"Dataset encoded. training time {encode_end - encode_start}")


def detect_face(face_input):
    face_cascade = cv2.CascadeClassifier(
        "haarcascade_frontalface_default.xml")
    face_detect = face_cascade.detectMultiScale(
        face_input, scaleFactor=1.1, minNeighbors=5)
    name = str()
    if(len(face_detect) == 0):
        return None, None
    for (x, y, w, h) in face_detect:
        detected_face = face_input[int(y):int(y+h), int(x):int(x+w)]
        detected_face = cv2.resize(
            detected_face, (224, 224))
        img_px = image.img_to_array(detected_face)
        img_px = np.expand_dims(img_px, axis=0)
        img_px /= 127.5
        img_px -= 1
        face_vector = vgg_descriptor.predict(img_px)[0, :]
        found = 0
        for face in faces:
            person_name = face
            representation = faces[face]
            similarity = find_cos_sim(representation, face_vector)
            if(similarity <= 0.4):
                first, last, trsh = person_name.split("_", 2)
                name = f"{first} {last}"
                cv2.putText(face_input, name, (x, y+15),
                            cv2.FONT_HERSHEY_DUPLEX, 1, blue_color, 1)
                cv2.rectangle(face_input, (x, y), (x+w, y+h), pink_color, 2)
                found = 1
                return face_input, name
        if(found == 0):
            not_recognized = "not recognized"
            cv2.putText(face_input, not_recognized, (int(w), int(h)),
                        cv2.FONT_HERSHEY_DUPLEX, 1, blue_color, 1)
            return face_input, not_recognized


test_data = "PATH_TO_TEST_IMAGES_OF_FACES"
times = []


def test_faces(path_to_test_data):
    images = os.listdir(path_to_test_data)
    name = str()
    for image in images:
        path_of_test_img = path_to_test_data + "/" + image
        print(image)
        if image.startswith("."):
            continue
        img = cv2.imread(path_of_test_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predict_start = time.time()
        prediction, name = detect_face(img)
        predict_end = time.time()
        predict_time = predict_end - predict_start
        print(f"Predicted in : {predict_time}")
        times.append(predict_time)
        plt.imshow(prediction)
        plt.title(name)
        plt.show()


test_faces(test_data)
