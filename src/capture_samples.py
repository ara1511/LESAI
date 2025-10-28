import sys
import os
# Añadir la carpeta raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import mediapipe as mp
from utils.constants import ACTIONS, NO_SEQUENCES, SEQUENCE_LENGTH, KEYPOINTS_PATH

# Crear la carpeta de keypoints si no existe
os.makedirs(KEYPOINTS_PATH, exist_ok=True)

# Inicializar MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_keypoints(results):
    """Extrae keypoints de manos y rostro."""
    keypoints = []
    
    # Mano izquierda (63)
    if results.left_hand_landmarks:
        for landmark in results.left_hand_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z])
    else:
        keypoints.extend([0.0] * 63)
    
    # Mano derecha (63)
    if results.right_hand_landmarks:
        for landmark in results.right_hand_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z])
    else:
        keypoints.extend([0.0] * 63)
    
    # Rostro (solo 50 puntos clave)
    if results.face_landmarks:
        selected_indices = list(range(0, 10)) + list(range(150, 160)) + list(range(250, 260))
        for idx in selected_indices:
            if idx < len(results.face_landmarks.landmark):
                landmark = results.face_landmarks.landmark[idx]
                keypoints.extend([landmark.x, landmark.y, landmark.z])
            else:
                keypoints.extend([0.0, 0.0, 0.0])
    else:
        keypoints.extend([0.0] * 150)
    
    return np.array(keypoints, dtype=np.float32)

def there_hand(results):
    """Verifica si hay al menos una mano en el frame."""
    return results.left_hand_landmarks is not None or results.right_hand_landmarks is not None

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: No se puede acceder a la cámara.")
        return

    # Configurar ventana (tamaño normal, no fullscreen)
    cv2.namedWindow('Recaptura Visual - LESAI', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Recaptura Visual - LESAI', 800, 600)

    print(f"\n🎯 ¡RECAPTURANDO TODAS LAS MUESTRAS DESDE CERO!")
    print(f"📝 Palabras: {ACTIONS}")
    print(f"📊 Total de muestras: {len(ACTIONS) * NO_SEQUENCES}")
    print("ℹ️  El sistema se activará automáticamente al detectar tus manos.")
    print("ℹ️  Presiona 'q' o 'ESC' en cualquier momento para salir.\n")

    for action_idx, action in enumerate(ACTIONS):
        action_path = os.path.join(KEYPOINTS_PATH, action)
        os.makedirs(action_path, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"🔄 RECAPTURANDO: '{action.upper()}' ({action_idx+1}/{len(ACTIONS)})")
        print(f"📸 Se capturarán {NO_SEQUENCES} muestras NUEVAS.")
        print(f"{'='*60}")
        
        captured_samples = 0
        while captured_samples < NO_SEQUENCES:
            print(f"\n➡️  Capturando muestra {captured_samples+1}/{NO_SEQUENCES} para '{action}'")
            print("   👁️  Muestra tus manos para comenzar a grabar...")
            
            sequence = []
            recording = False
            no_hand_count = 0
            NO_HAND_THRESHOLD = 60

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Procesar con Holistic
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame_rgb)

                # Dibujar solo manos y rostro (sin cuerpo)
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, 
                        results.left_hand_landmarks, 
                        mp_holistic.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                    )
                
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, 
                        results.right_hand_landmarks, 
                        mp_holistic.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                    )
                
                if results.face_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, 
                        results.face_landmarks, 
                        mp_holistic.FACEMESH_CONTOURS,
                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1)
                    )

                # Lógica de grabación automática
                if there_hand(results):
                    no_hand_count = 0
                    if not recording:
                        print("   🎥 ¡Manos detectadas! Comenzando grabación...")
                        recording = True
                        sequence = []
                    
                    if recording:
                        kp = extract_keypoints(results)
                        sequence.append(kp)
                        
                        cv2.putText(frame, f"GRABANDO: {len(sequence)}/{SEQUENCE_LENGTH}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        if len(sequence) >= SEQUENCE_LENGTH:
                            sequence_path = os.path.join(action_path, f"{captured_samples}.npy")
                            np.save(sequence_path, np.array(sequence))
                            captured_samples += 1
                            print(f"   ✅ Muestra {captured_samples} guardada.")
                            break
                
                else:
                    no_hand_count += 1
                    if no_hand_count > NO_HAND_THRESHOLD:
                        if recording:
                            print("   ⚠️  Grabación cancelada: No se detectaron manos.")
                        recording = False
                        sequence = []
                        break

                # Mostrar estado en la parte inferior
                status = "LISTO" if not recording else "GRABANDO"
                color = (0, 255, 0) if not recording else (0, 0, 255)
                cv2.putText(frame, f"Estado: {status}", (10, frame.shape[0] - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, f"'{action}' - Muestra {captured_samples+1}/{NO_SEQUENCES}", 
                           (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow('Recaptura Visual - LESAI', frame)
                
                # ✅ Salir con 'q' o 'ESC'
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 27 es el código de la tecla Esc
                    cap.release()
                    cv2.destroyAllWindows()
                    print("\n⏹️  Recaptura cancelada por el usuario.")
                    return

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n🎉 ¡RECAPTURA COMPLETADA DESDE CERO!")
    print(f"🔄 Todos los datos han sido reemplazados en: {KEYPOINTS_PATH}")

if __name__ == "__main__":
    main()