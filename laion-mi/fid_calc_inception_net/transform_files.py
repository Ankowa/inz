# script that extracts the data from subdirectories and puts them into a single directory

import os
from tqdm import tqdm
from PIL import Image

import cv2
import numpy as np

import threading

import torch

from torchvision import transforms

# path to the directory where the subdirectories are located
path = "non-members"

# path to the directory where the files will be moved
output_path_first = "non_members_small_first"

output_path_second = "non_members_small_second"

# get all subdirectories

subdirs = sorted([x[0] for x in os.walk(path)])

# create the output directory if it does not exist

if not os.path.exists(output_path_first):
    os.mkdir(output_path_first)

if not os.path.exists(output_path_second):
    os.mkdir(output_path_second)


def transform(img: Image) -> Image:
    transform = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToPILImage(),
        ]
    )
    return transform(img)


def load(img_filename) -> Image:
    try:
        return Image.open(img_filename).convert("RGB")
    except Exception as e:
        image = cv2.imread(img_filename)
        image = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )  # can fail if bytes are e.g. HTML for some reason
        image = Image.fromarray(image)
        return image


def save_as_jpg(img_filename, transformed_img: Image):
    transformed_img.save(img_filename, "JPEG")


def save_as_png(img_filename, transformed_img: Image):
    transformed_img.save(img_filename, "PNG")


def save(img_filename, transformed_img: Image):
    if img_filename.endswith(".jpg"):
        save_as_jpg(img_filename, transformed_img)
    elif img_filename.endswith(".png"):
        save_as_png(img_filename, transformed_img)
    else:
        raise Exception("Unknown file format")


# use multithreading to load and transform the images
# create thread class


class Thread(threading.Thread):
    def __init__(self, img_filenames, target_directory):
        threading.Thread.__init__(self)
        self.img_filenames = img_filenames
        self.target_directory = target_directory
        self.images_saved = 0

    def run(self):
        for img_filename in tqdm(self.img_filenames):
            try:
                image = load(img_filename)
                transformed_img = transform(image)
                save(
                    os.path.join(self.target_directory, os.path.basename(img_filename)),
                    transformed_img,
                )
                self.images_saved += 1
            except Exception as e:
                print(e)
                continue


limit = 20_000

for idx, subdir in enumerate(subdirs):
    if not idx:
        continue
    files = os.listdir(subdir)
    img_filenames = [
        os.path.join(subdir, file) for file in files if "png" in file or "jpg" in file
    ]

    # split the files into two parts
    first_part = img_filenames[: len(img_filenames) // 2]
    second_part = img_filenames[len(img_filenames) // 2 :]

    # create threads
    thread1 = Thread(first_part, output_path_first)
    thread2 = Thread(second_part, output_path_second)

    # start threads
    thread1.start()
    thread2.start()

    # wait for threads to finish
    thread1.join()
    thread2.join()

    print(f"Images saved in {output_path_first}: {thread1.images_saved}")
    print(f"Images saved in {output_path_second}: {thread2.images_saved}")

    # stop if the limit is reached

    limit -= thread1.images_saved + thread2.images_saved

    if limit <= 0:
        break

print(len(subdirs))
