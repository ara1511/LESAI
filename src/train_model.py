import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from utils.constants import ACTIONS, KEYPOINTS_PATH, SEQUENCE_LENGTH, KEYPOINT_DIM, MODELS_PATH

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
        action_path = os.path.join(KEYPOINTS_PATH, action)
        if not os.path.exists(action_path):
            print(f"Advertencia: No se encontró la carpeta para la acción '{action}'.")
            continue

        seq_files = [f for f in os.listdir(action_path) if f.endswith('.npy')]
        print(f"Cargando {len(seq_files)} muestras para la acción '{action}'.")

        for seq_file in seq_files:
            try:
                keypoints = np.load(os.path.join(action_path, seq_file))
                # Verificar que la forma sea correcta (10, 276)
                if keypoints.shape == (SEQUENCE_LENGTH, KEYPOINT_DIM):
                    X.append(keypoints)
                    y.append(label_map[action])
                else:
                    print(f"Advertencia: Forma incorrecta en {action}/{seq_file}: {keypoints.shape}")
            except Exception as e:
                print(f"Error cargando {action}/{seq_file}: {e}")

    if len(X) == 0:
        raise ValueError("No se cargaron datos. Asegúrate de haber ejecutado 'capture_samples.py'.")

    X = np.array(X)
    y = to_categorical(y, num_classes=len(ACTIONS)).astype(int)
    return X, y

if __name__ == "__main__":
    print("Cargando datos de entrenamiento...")
    X, y = load_data()
    print(f"Datos cargados: {X.shape[0]} muestras, {X.shape[1]} frames, {X.shape[2]} keypoints")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Creando y entrenando el modelo...")
    model = create_model((SEQUENCE_LENGTH, KEYPOINT_DIM), len(ACTIONS))
    history = model.fit(
        X_train, y_train, 
        epochs=100, 
        validation_data=(X_test, y_test), 
        batch_size=8
    )

    # Guardar el modelo
    os.makedirs(MODELS_PATH, exist_ok=True)
    model.save(f'{MODELS_PATH}/actions.keras')
    print(f"Modelo guardado en {MODELS_PATH}/actions.keras")

    # Guardar el mapeo de etiquetas
    label_map_for_saving = {str(idx): action for idx, action in enumerate(ACTIONS)}
    with open(f'{MODELS_PATH}/label_map.json', 'w') as f:
        json.dump(label_map_for_saving, f, indent=4)
    print(f"Mapeo de etiquetas guardado en {MODELS_PATH}/label_map.json")

    # Imprimir resumen
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nPrecisión en el conjunto de prueba: {test_acc:.2%}")