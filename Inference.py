from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse

my_parser = argparse.ArgumentParser()
my_parser.add_argument('--input_photo', type=str ,help="Enter path of your image")
my_parser.add_argument('--input_model', type=str ,help="Enter path of your model")

args = my_parser.parse_args()

model = load_model(args.input_model)

width = height = 256

img = cv2.imread(args.input_photo)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img / 255.0
img = cv2.resize(img, (width, height)).astype(np.float32)
img = img.reshape(1, width, height, 3)

generate = model(img,training=True)
generate = np.squeeze(generate, axis=0)
generate = np.array((generate +1) *127.5).astype(np.uint8)

plt.imsave("Output.png",generate)