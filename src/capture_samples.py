"""
Script para capturar muestras de señas directamente como keypoints.
Este script es interactivo y te guía a través del proceso para hacerlo menos tedioso.
"""
import os
import cv2
import numpy as np
import mediapipe as mp
from utils.constants import ACTIONS, NO_SEQUENCES, SEQUENCE_LENGTH, KEYPOINTS_PATH

# Crear la carpeta de keypoints si no existe
os.makedirs(KEYPOINTS_PATH, exist_ok=True)

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_keypoints(results):
    """Extrae 126 keypoints de los resultados de MediaPipe."""
    keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
    # Rellenar o recortar a 126
    keypoints = keypoints[:126]
    while len(keypoints) < 126:
        keypoints.append(0.0)
    return np.array(keypoints, dtype=np.float32)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se puede acceder a la cámara.")
        return

    for action in ACTIONS:
        # Crear la carpeta para la acción
        action_path = os.path.join(KEYPOINTS_PATH, action)
        os.makedirs(action_path, exist_ok=True)

        print(f'\n' + '='*50)
        print(f'PREPÁRATE PARA LA SEÑA: "{action.upper()}"')
        print(f'Capturaremos {NO_SEQUENCES} muestras.')
        print(f'Presiona la BARRA ESPACIADORA para comenzar cada muestra.')
        print(f'Presiona "q" para salir en cualquier momento.')
        print('='*50)

        for sequence in range(NO_SEQUENCES):
            print(f"\nMuestra {sequence+1}/{NO_SEQUENCES} para '{action}' - ¡LISTO! Presiona ESPACIO para grabar...")

            # Esperar a que el usuario presione ESPACIO
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                # Mostrar instrucciones en el frame
                cv2.putText(frame, f'PREPARANDO: "{action}" ({sequence+1}/{NO_SEQUENCES})', 
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, 'Presiona ESPACIO para grabar', 
                            (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('Captura de Señas - Presiona ESPACIO', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):  # Barra espaciadora
                    break
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # Grabar la secuencia
            keypoints_sequence = []
            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret:
                    break

                # Procesar el frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                # Dibujar los landmarks en el frame para feedback visual
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )

                # Extraer y guardar los keypoints
                keypoints = extract_keypoints(results)
                keypoints_sequence.append(keypoints)

                # Mostrar feedback en el frame
                cv2.putText(frame, f'GRABANDO: {action}', (15, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Frame: {frame_num+1}/{SEQUENCE_LENGTH}', (15, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Captura de Señas - GRABANDO', frame)

                # Pequeña pausa para que la seña sea fluida
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    break

            # Guardar la secuencia de keypoints
            sequence_path = os.path.join(action_path, f"{sequence}.npy")
            np.save(sequence_path, np.array(keypoints_sequence))

            print(f"  -> Muestra {sequence+1} guardada en {sequence_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n¡Captura completada con éxito!")
    print(f"Los datos se han guardado en: {KEYPOINTS_PATH}")

if __name__ == "__main__":
    main()