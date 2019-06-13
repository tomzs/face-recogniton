import matplotlib.pyplot as plt
import numpy as np
import os
import cv2



faces_to_match_against = "folder_of_face_images_to_flip"


directories = os.listdir(faces_to_match_against)
for directory in directories:
    if directory == "desktop.ini":
        continue
    person_dir = faces_to_match_against + "/" + directory
    person_images = os.listdir(person_dir)
    for n_image in person_images:
        name, ext = n_image.split(".")
        # Ignore hidden files
        if n_image.startswith("."):
            continue
        file_name = person_dir + "/" + n_image

        new_name = person_dir + "/" + name + "_flipped." + ext
        print(file_name)
        print(new_name)
        print(ext)
        im = np.fliplr(plt.imread(file_name))
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        cv2.imwrite(new_name, rgb)
