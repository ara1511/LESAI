import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from utils.constants import ACTIONS, KEYPOINTS_PATH, SEQUENCE_LENGTH, KEYPOINT_DIM

def create_model(input_shape, num_classes):
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape),
        LSTM(128, return_sequences=True, activation='relu'),
        LSTM(64, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_data():
    X, y = [], []
    label_map = {label: idx for idx, label in enumerate(ACTIONS)}

    for action in ACTIONS:
        for seq in range(5):  # Solo 5 muestras por acción
            try:
                keypoints = np.load(os.path.join(KEYPOINTS_PATH, action, f"{seq}.npy"))
                X.append(keypoints)
                y.append(label_map[action])
            except Exception as e:
                print(f"Error cargando {action}/{seq}: {e}")

    X = np.array(X)
    y = to_categorical(y).astype(int)
    return X, y

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = create_model((SEQUENCE_LENGTH, KEYPOINT_DIM), len(ACTIONS))
    model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=1)  # Menos epochs

    model.save('models/actions.keras')

    # Guarda el mapeo de etiquetas
    with open('models/label_map.json', 'w') as f:
        json.dump({str(v): k for k, v in {label: num for num, label in enumerate(ACTIONS)}.items()}, f)