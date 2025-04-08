import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from sklearn.model_selection import train_test_split

# Load and preprocess dataset
def load_images(folder, img_size=(128, 128)):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)
    return np.array(images) / 255.0  # Normalize

# Paths to dataset
train_img_path = "F:\\open lab\\datasets\\data_road_224\\training\\image" # Path to traffic sign images
train_mask_path = "F:\\open lab\\datasets\\data_road_224\\training\\gt_image"  # Path to the corresponding ground truth masks

# Load images and masks
X = load_images(train_img_path)
Y = load_images(train_mask_path, img_size=(128, 128))[:, :, :, 0:1]  # Convert masks to grayscale

# Split dataset
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# U-Net model
def unet():
    inputs = Input(shape=(128, 128, 3))
    
    c1 = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    c1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation="relu", padding="same")(c1)
    c2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(128, (3, 3), activation="relu", padding="same")(c2)
    c3 = MaxPooling2D((2, 2))(c3)

    u1 = UpSampling2D((2, 2))(c3)
    u1 = Conv2D(64, (3, 3), activation="relu", padding="same")(u1)

    u2 = UpSampling2D((2, 2))(u1)
    u2 = Conv2D(32, (3, 3), activation="relu", padding="same")(u2)

    u3 = UpSampling2D((2, 2))(u2)
    outputs = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(u3)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Train the model
model = unet()
model.fit(X_train, Y_train, epochs=50, batch_size=11, validation_data=(X_val, Y_val))

# Test the model with a new image
test_img_path = "F:\\open lab\\datasets\\data_road_224\\testing\\image_2\\um_000013.png" # Test image path
assert os.path.exists(test_img_path), f"Test image path {test_img_path} is incorrect!"

test_img = cv2.imread(test_img_path)
if test_img is None:
    raise ValueError(f"Failed to load image: {test_img_path}")
    
test_img = cv2.resize(test_img, (128, 128)) / 255.0
test_img = np.expand_dims(test_img, axis=0)

prediction = model.predict(test_img)[0]
prediction = (prediction > 0.5).astype(np.uint8)  # Convert to binary mask

# Show prediction result
cv2.imshow("Traffic Sign Prediction", prediction * 255)  # Convert to 0-255 for display
cv2.waitKey(0)
cv2.destroyAllWindows()