import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random

# Define data directory
dir = "train/"
train_emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Read all images
img_size_height = 224 # ImageNet image dimensions are 224 x 224
img_size_width = 224 # ImageNet image dimensions are 224 x 224

def create_training_data():
    training_data = []
    
    for emotion in train_emotions:
        path = os.path.join(dir, emotion)
        emotion_categories = train_emotions.index(emotion) # Convert emotion into a number (0-6)
        
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))
                resize_arr = cv2.resize(img_arr, (img_size_height, img_size_width))
                training_data.append([resize_arr, emotion_categories]) # Image and emotion (0-6)
            
            except Exception as e:
                pass
    
    return training_data

training_data = create_training_data()
random.shuffle(training_data)

print("Training data established!")

def create_numpy_array(training_data):
    X = [] # Features
    y = [] # Labels

    for features, labels in training_data:
        X.append(features)
        y.append(labels)

    X = np.array(X).reshape(-1, img_size_height, img_size_width, 3) # batch_size, height, width, channels format for DL
                                                                 # (-1): total number of samples in X (# of images in training)
                                                                 # (3): three color channels (R, G, B)

    # Data normalization
    X = X / 255.0 # 0 - white, ... , 225 - black, so normalize all by dividing by max (255)
    
    y = np.array(y)

    return X, y

X, y = create_numpy_array(training_data)
print("Arrays established!")

# Apply Deep Learning Model via Transfer Learning

def load_pre_trained_model():
    model = tf.keras.applications.MobileNetV2(weights = 'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5') # Pre-trained model! 
    #v3 is also available
    # #model.summary()
    base_input = model.input
    base_output = model.layers[-2].output # global_average_pooling2d layer from pre-trained model

    return base_input, base_output

def new_model(base_input, base_output):
    n_minus_1_layer = tf.keras.layers.Dense(128)(base_output) # New layer (comes after global_average_pooling2d layer) with 128 nodes
    n_minus_1_output = tf.keras.layers.Activation('relu')(n_minus_1_layer) # Activation
    final_layer = tf.keras.layers.Dense(64)(n_minus_1_output)
    final_activation_layer = tf.keras.layers.Activation('relu')(final_layer)
    final_output = tf.keras.layers.Dense(7, activation = 'softmax')(final_activation_layer) # 7 emotions
    model_with_new_layers = keras.Model(inputs = base_input, outputs = final_output)
    model_with_new_layers.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy']) # Choose this loss function since y-labels 
                                                                                                                        # are not 1-hot encoded

    return model_with_new_layers

base_input, base_output = load_pre_trained_model()
model_with_new_layers = new_model(base_input, base_output)

print("Model architecture established!")

# Fit model
if os.path.exists("Final_model_trained_vFinal.h5"):
    model_with_new_layers = tf.keras.models.load_model("Final_model_trained_vFinal.h5")
    print("Model loaded.")

else:
    model_with_new_layers.fit(X, y, epochs = 30)
    print("Training Completed...")
    model_with_new_layers.save("Final_model_trained_vFinal.h5")
    print("Model saved.")

# Apply model
model_with_new_layers = tf.keras.models.load_model("Final_model_trained_vFinal.h5")

# Real time emotion detection
if os.path.exists("haarcascade_frontalface_default.xml"):
    font_scale = 2
    font = cv2.FONT_HERSHEY_PLAIN

    # Configure facial recognition rectangle
    rectangle_hex = (255, 255, 255)
    img = np.zeros((500, 500))
    text = "Some text in box!!!!"

    (text_width, text_height) = cv2.getTextSize(text, font, fontScale = font_scale, thickness = 1)[0]
    text_x_offset = 10
    text_y_offset = img.shape[0] - 25

    box_coordiate = ((text_x_offset, text_y_offset), (text_x_offset + text_width + 2, text_y_offset - text_height - 2))
    cv2.rectangle(img, box_coordiate[0], box_coordiate[1], rectangle_hex, cv2.FILLED)
    cv2.putText(img, text, (text_x_offset, text_y_offset), font, fontScale = font_scale, color =(0, 0, 0), thickness = 1)

    capture = cv2.VideoCapture(1)
    if not capture.isOpened():
        capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise IOError("Webcam cannot be opened...")
    
    while True:
        ret, frame = capture.read()
        faceCas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCas.detectMultiScale(gray, 1.1, 4)

        face_roi = None

        for x, y, w, h in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (225, 0, 0), 2)
            faces_final = faceCas.detectMultiScale(roi_gray)

            if len(faces_final) == 0:
                print("Face not detected.")
            
            else:
                for (ex, ey, ew, eh) in faces_final:
                    face_roi = roi_color[ey:ey+eh, ex:ex+ew]

        if face_roi is not None:
            final_img = cv2.resize(face_roi, (224, 224))
            final_img = np.expand_dims(final_img, axis=0)
            final_img = final_img / 255.0  # Normalization

            font = cv2.FONT_HERSHEY_SIMPLEX

            preds = model_with_new_layers.predict(final_img)
            print(preds)

            font_scale = 1.5
            font = cv2.FONT_HERSHEY_PLAIN

            if (np.argmax(preds) == 0):
                label = "Angry"

                x1, y1, w1, h1 = 0, 0, 175, 175
                cv2.rectangle(frame, (x1, x1,), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                cv2.putText(frame, label, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, label, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
            
            elif (np.argmax(preds) == 1):
                label = "Disgust"

                x1, y1, w1, h1 = 0, 0, 175, 175
                cv2.rectangle(frame, (x1, x1,), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                cv2.putText(frame, label, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, label, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))

            elif (np.argmax(preds) == 2):
                label = "Fear"

                x1, y1, w1, h1 = 0, 0, 175, 175
                cv2.rectangle(frame, (x1, x1,), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                cv2.putText(frame, label, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, label, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))

            elif (np.argmax(preds) == 3):
                label = "Happy"

                x1, y1, w1, h1 = 0, 0, 175, 175
                cv2.rectangle(frame, (x1, x1,), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                cv2.putText(frame, label, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, label, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))

            elif (np.argmax(preds) == 4):
                label = "Neutral"

                x1, y1, w1, h1 = 0, 0, 175, 175
                cv2.rectangle(frame, (x1, x1,), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                cv2.putText(frame, label, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, label, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
            
            elif (np.argmax(preds) == 5):
                label = "Sad"

                x1, y1, w1, h1 = 0, 0, 175, 175
                cv2.rectangle(frame, (x1, x1,), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                cv2.putText(frame, label, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, label, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
            
            elif (np.argmax(preds) == 6):
                label = "Surprised"

                x1, y1, w1, h1 = 0, 0, 175, 175
                cv2.rectangle(frame, (x1, x1,), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                cv2.putText(frame, label, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, label, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))

            else:
                label = "Neutral"

                x1, y1, w1, h1 = 0, 0, 175, 175
                cv2.rectangle(frame, (x1, x1,), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                cv2.putText(frame, label, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, label, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))

            cv2.imshow("Face Emotion Recognition", frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    
    capture.release()
    cv2.destroyAllWindows()

else:
    raise IOError("Module not found. Please download the haarcascade_frontalface_default.xml file from Kaggle")