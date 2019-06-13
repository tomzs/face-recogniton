import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time

dataset_path = os.listdir(
    "PATH_TO_FOLDERS_OF_FACE_IMAGES")
test_data = "PATH_TO_TEST_FACE_IMAGES"
persons = []
prec = []
training_start = time.time()

for count, dir_name in enumerate(dataset_path, start=1):
    person_map = {"id": count, "name": dir_name}
    persons.append(person_map)


def detect_face(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        "haarcascade_frontalface_default.xml")
    detected_face = face_cascade.detectMultiScale(
        grayscale, scaleFactor=1.1, minNeighbors=7)

    if(len(detected_face) == 0):
        return None, None
    x, y, w, h = detected_face[0]
    return grayscale[y:y+w, x:x+h], detected_face[0]


def prep_training_data(path_to_data):
    directories = os.listdir(path_to_data)
    faces = []
    labels = []
    label = 0
    for directory in directories:
        label_id = [i['id'] for i in persons if i['name'] == directory]
        label = int(label_id[0])
        person_dir = path_to_data + "/" + directory
        person_images = os.listdir(person_dir)

        for n_image in person_images:
            if n_image.startswith("."):
                continue
            path_of_image = person_dir + "/" + n_image
            image = cv2.imread(path_of_image)
            img = cv2.resize(image, (250, 250))

            face, bounding_box = detect_face(img)
            if face is not None:
                face = cv2.resize(face, (100, 100))
                faces.append(face)
                labels.append(label)
    return faces, labels


faces, labels = prep_training_data(dataset_path)
eigenfaces_face_recognition = cv2.face.EigenFaceRecognizer_create()
eigenfaces_face_recognition.train(faces, np.array(labels))

training_end = time.time()


def draw_rect(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (236, 110, 173), 2)


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                0.7, (52, 148, 230), 2)


def predict(face_image):
    prediction_start = time.time()
    img = face_image.copy()
    face, rect = detect_face(img)
    face = cv2.resize(face, (100, 100))
    label, distance = eigenfaces_face_recognition.predict(face)
    print(f"RECOGNIZED LABEL {label} with {round(distance, 2)}%")
    get_from_dict = [i['name'] for i in persons if i['id'] == label]
    person = str(get_from_dict[0])
    draw_rect(img, rect)
    draw_text(img, person, rect[0], rect[1]-5)
    prediction_end = time.time()
    print(f"Predicted in {prediction_end - prediction_start} s ")
    prec.append(str(prediction_end - prediction_start) + "\n")
    return img


def test_faces(path_to_test_data):
    images = os.listdir(path_to_test_data)
    for image in images:
        path_of_test_img = path_to_test_data + "/" + image
        print(image)
        if image.startswith("."):
            continue
        img = cv2.imread(path_of_test_img)
        print(f"Test data shape: {img.shape}")
        prediction = predict(img)
        cv2.imshow(image, prediction)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


print(f"Trained in: {training_end - training_start}")
print("Starting...")

test_faces(test_data)
