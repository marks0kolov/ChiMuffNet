#!/usr/bin/env python3

import sys
import os
import numpy as np
import cv2
import onnxruntime as rt

if len(sys.argv) != 4:
    print(f"usage: {sys.argv[0]} model.onnx image_or_folder labels.txt")
    sys.exit(1)

model_path, path, labels_path = sys.argv[1], sys.argv[2], sys.argv[3]

if os.path.isdir(path):
    files = os.listdir(path)
    jpgs = [f for f in files if f.lower().endswith('.jpg')]
    if len(jpgs) != len(files):
        print("error: folder must contain only .jpg images")
        sys.exit(1)
    image_paths = [os.path.join(path, f) for f in jpgs]
else:
    image_paths = [path]

sess = rt.InferenceSession(model_path, providers=['CPUExecutionProvider'])
inp = sess.get_inputs()[0]
h, w = inp.shape[2] or 224, inp.shape[3] or 224

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

with open(labels_path) as f:
    labels = [l.strip() for l in f]

for img_path in image_paths:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h)).astype(np.float32) / 255.0
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))[None, ...]

    outputs = sess.run(None, {inp.name: img})
    probs = outputs[0][0]
    idx = int(np.argmax(probs))
    conf = probs[idx] * 100

    print(f"{os.path.basename(img_path)} -> {labels[idx]} {conf:.2f}%")
