import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
import numpy as np
import os
from utils.constants import ACTIONS, DATA_PATH, KEYPOINTS_PATH, SEQUENCE_LENGTH

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

def extract_keypoints(results):
    keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
    # Rellenar con ceros si no hay manos detectadas
    while len(keypoints) < 126:
        keypoints.append(0)
    return np.array(keypoints[:126])

def process_all_sequences():
    for action in ACTIONS:
        for seq in range(5):  # Solo 5 muestras por acción
            seq_path = os.path.join(DATA_PATH, action, str(seq))
            keypoints_list = []

            for frame_num in range(SEQUENCE_LENGTH):
                frame_path = os.path.join(seq_path, f"{frame_num}.jpg")
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                kp = extract_keypoints(results)
                keypoints_list.append(kp)

            save_path = os.path.join(KEYPOINTS_PATH, action, f"{seq}.npy")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, np.array(keypoints_list))

if __name__ == "__main__":
    process_all_sequences()