import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import os
from utils.constants import ACTIONS, DATA_PATH, SEQUENCE_LENGTH

def capture_samples():
    for action in ACTIONS:
        for sequence in range(5):  # Solo 5 muestras por acción (más rápido)
            seq_path = os.path.join(DATA_PATH, action, str(sequence))
            os.makedirs(seq_path, exist_ok=True)

            cap = cv2.VideoCapture(0)
            print(f"Capturando: {action} - Secuencia {sequence}")

            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret:
                    break

                cv2.putText(frame, f'{action} - Seq {sequence} - Frame {frame_num}', 
                            (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow('Captura', frame)

                frame_path = os.path.join(seq_path, f"{frame_num}.jpg")
                cv2.imwrite(frame_path, frame)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_samples()