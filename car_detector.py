# python3
#
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example using TF Lite to detect objects with the Raspberry Pi camera."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image


class CarDetector:
    def __init__(self, model_path, label_path):
        self.model_file = "mobilenetv2_parking_lot_free_busy_nr_data_same.h5"
        self.input_size = 160
        self.model = tf.keras.models.load_model(self.model_file)

    def detect2(self, img):
        img = cv2.resize(img, dsize=(self.input_size, self.input_size))
        img = img[:, :, [2, 1, 0]]  # BGR -> RGB
        im = np.reshape(img, [1, self.input_size, self.input_size, 3])
        result = self.model.predict(im)
        return result[0]

