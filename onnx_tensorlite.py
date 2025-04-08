# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 12:59:30 2025

@author: sgpdh
"""

import tf2onnx
import onnx

onnx_model_path = r"D:\SEM 6\Open lab\project\roboflow_dataset1\traffic_light_model_v2.onnx"
saved_model_dir = r"D:\SEM 6\Open lab\project\saved_model"

# Load the ONNX model
onnx_model = onnx.load(onnx_model_path)

# Convert to TensorFlow
tf_rep = tf2onnx.convert.from_onnx(onnx_model, output_path=saved_model_dir)

print("✅ Model converted to TensorFlow SavedModel!")
