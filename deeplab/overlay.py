#!/usr/bin/env python3
"""
@Filename:    overlay.py
@Author:      dulanj
@Time:        02/10/2021 19:22
"""
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from deeplab.dataset import read_image
from deeplab.inference import inference
from deeplab.params import DATASET_DIR, PRED_OUTPUT
from scipy.io import loadmat
import os
import matplotlib.patches as mpatches
# Loading the Colormap
colormap = loadmat(os.path.join(DATASET_DIR, "human_colormap.mat"))["colormap"]
colormap = colormap * 100
colormap = colormap.astype(np.uint8)


def decode_segmentation_masks(mask, colormap, n_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def get_patches(colormap, n_classes):
    labels = [
        'Background', 'Hat', 'Hair', 'Glove',
        'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
        'Socks', 'Pants', 'Tosor-skin', 'Scarf',
        'Skirt', 'Face', 'Left-arm', 'Right-arm',
        'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    ## create patches as legend
    return [mpatches.Patch(color=[colormap[i, 0], colormap[i, 1], colormap[i, 2]], label=labels[i]) for i in range(n_classes)]


def get_overlay(image, colored_mask):
    image = tf.keras.preprocessing.image.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
    return overlay


def plot_samples_matplotlib(display_list, name, figsize=(5, 3)):
    fig, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            axes[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        else:
            axes[i].imshow(display_list[i])
    fig.legend(handles=get_patches(colormap / np.max(colormap), 20), loc=4, borderaxespad=0., ncol=5, fontsize=25)
    plt.show()
    plt.savefig(PRED_OUTPUT + '/figures/' + name)


def plot_predictions(images_list, model):
    pred_list = []
    for image_file in images_list:
        image_tensor = read_image(image_file)
        prediction_mask = inference(image_tensor=image_tensor, model=model)
        prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, 20)
        overlay = get_overlay(image_tensor, prediction_colormap)
        predict_image_list = [image_tensor, overlay, prediction_colormap]
        plot_samples_matplotlib(
            predict_image_list, os.path.basename(image_file), (18, 14)
        )
        pred_list.append(predict_image_list)
    return pred_list
