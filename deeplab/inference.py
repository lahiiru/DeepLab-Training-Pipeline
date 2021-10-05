#!/usr/bin/env python3
"""
@Filename:    inference.py
@Author:      dulanj
@Time:        02/10/2021 19:18
"""
import numpy as np
import tensorflow as tf
from deeplab.dataset import read_image

class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
  def __init__(self,
               y_true=None,
               y_pred=None,
               num_classes=None,
               name=None,
               dtype=None):
    super(UpdatedMeanIoU, self).__init__(num_classes = num_classes,name=name, dtype=dtype)

def load_model(model_path):
    deeplab_model = tf.keras.models.load_model(model_path, custom_objects={'UpdatedMeanIoU': UpdatedMeanIoU})
    return deeplab_model


def inference(model, image_path="", image_tensor=None):
    if image_tensor is None:
        image_tensor = read_image(image_path)
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions
