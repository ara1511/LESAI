import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import platform
import threading
import cv2
import numpy as np
import tensorflow as tf
import json
import pyttsx3
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QVBoxLayout, QWidget, QHBoxLayout,
                            QFrame, QGroupBox, QProgressBar, QTextEdit)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QIcon
from utils.constants import MODELS_PATH
import mediapipe as mp

# ------------------------- BOTÓN ANIMADO -------------------------
class AnimatedButton(QPushButton):
    def __init__(self, text, color="#4CAF50"):
        super().__init__(text)
        self.base_color = color
        self.setStyleSheet(self.get_base_style())
        
    def get_base_style(self):
        return f"""
            QPushButton {{
                background-color: {self.base_color};
                border: none;
                color: white;
                padding: 12px 24px;
                text-align: center;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
                min-width: 120px;
                min-height: 40px;
            }}
            QPushButton:hover {{
                background-color: {self.lighten_color(self.base_color)};
            }}
            QPushButton:pressed {{
                background-color: {self.darken_color(self.base_color)};
            }}
            QPushButton:disabled {{
                background-color: #cccccc;
                color: #666666;
            }}
        """
    
    def lighten_color(self, color):
        if color == "#4CAF50":
            return "#66BB6A"
        elif color == "#f44336":
            return "#EF5350"
        return color
    
    def darken_color(self, color):
        if color == "#4CAF50":
            return "#388E3C"
        elif color == "#f44336":
            return "#C62828"
        return color

# ------------------------- VENTANA PRINCIPAL -------------------------
class SignTranslatorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🖐️🤖 Comunicación Inclusiva con IA")
        self.setWindowIcon(QIcon('assets/icon.png'))
        self.setGeometry(100, 100, 1400, 900)
        
        # Tema oscuro
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; color: #ffffff; }
            QWidget { background-color: #1e1e1e; color: #ffffff; font-family: 'Segoe UI'; }
            QGroupBox { font-size: 14px; font-weight: bold; border: 2px solid #3c3c3c; 
                        border-radius: 8px; background-color: #2d2d2d; margin-top: 1ex; padding-top: 10px; }
            QLabel { color: #ffffff; }
            QTextEdit { background-color: #2d2d2d; border: 2px solid #3c3c3c; border-radius: 8px;
                        padding: 10px; font-size: 14px; color: #ffffff; }
            QProgressBar { border: 2px solid #3c3c3c; border-radius: 8px; background-color: #2d2d2d; }
            QProgressBar::chunk { background-color: #4CAF50; border-radius: 6px; }
        """)

        # Inicializar MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Cargar modelo y etiquetas
        self.model = tf.keras.models.load_model(f'{MODELS_PATH}/actions.keras')
        with open(f'{MODELS_PATH}/label_map.json', 'r') as f:
            self.label_map = json.load(f)

        # Variables
        self.sequence = []
        self.sentence = []
        self.translation_history = []
        self.threshold = 0.7
        self.cap = None
        self.current_confidence = 0.0

        self.setup_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    # ------------------------- UI -------------------------
    def setup_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setSpacing(20)
        self.main_layout.setContentsMargins(20, 20, 20, 20)

        # Panel izquierdo
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)

        # Cámara
        self.video_group = QGroupBox("📹 Cámara en Tiempo Real")
        self.video_layout = QVBoxLayout(self.video_group)
        self.video_label = QLabel("Cámara sin inicializar")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("color: #888888; font-size: 18px; font-weight: bold;")
        self.video_layout.addWidget(self.video_label)

        # Controles
        self.controls_group = QGroupBox("🎮 Controles")
        self.controls_layout = QHBoxLayout(self.controls_group)
        self.start_btn = AnimatedButton("▶️ Iniciar Traducción", "#4CAF50")
        self.start_btn.clicked.connect(self.start_translation)
        self.stop_btn = AnimatedButton("⏹️ Detener", "#f44336")
        self.stop_btn.clicked.connect(self.stop_translation)
        self.stop_btn.setEnabled(False)
        self.controls_layout.addWidget(self.start_btn)
        self.controls_layout.addWidget(self.stop_btn)
        self.left_layout.addWidget(self.video_group)
        self.left_layout.addWidget(self.controls_group)

        # Panel derecho
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)

        # Traducción actual
        self.translation_group = QGroupBox("🔤 Traducción Actual")
        self.translation_layout = QVBoxLayout(self.translation_group)
        self.current_translation_label = QLabel("Esperando señas...")
        self.current_translation_label.setAlignment(Qt.AlignCenter)
        self.current_translation_label.setStyleSheet("""
            QLabel {
                background-color: #2d2d2d; border: 2px solid #4CAF50;
                border-radius: 12px; padding: 20px; font-size: 28px;
                font-weight: bold; color: #4CAF50; min-height: 80px;
            }
        """)
        self.translation_layout.addWidget(self.current_translation_label)
        self.confidence_label = QLabel("Confianza: 0%")
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.translation_layout.addWidget(self.confidence_label)
        self.translation_layout.addWidget(self.confidence_bar)

        # Historial
        self.history_group = QGroupBox("📝 Historial de Traducciones")
        self.history_layout = QVBoxLayout(self.history_group)
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        self.history_text.setMaximumHeight(200)
        self.history_layout.addWidget(self.history_text)

        # Estadísticas
        self.stats_group = QGroupBox("📊 Estadísticas")
        self.stats_layout = QVBoxLayout(self.stats_group)
        self.words_translated_label = QLabel("Palabras traducidas: 0")
        self.avg_confidence_label = QLabel("Confianza promedio: 0%")
        for label in [self.words_translated_label, self.avg_confidence_label]:
            label.setStyleSheet("font-size: 14px; padding: 5px; color: #cccccc;")
            self.stats_layout.addWidget(label)

        self.right_layout.addWidget(self.translation_group)
        self.right_layout.addWidget(self.history_group)
        self.right_layout.addWidget(self.stats_group)
        self.main_layout.addWidget(self.left_panel, 2)
        self.main_layout.addWidget(self.right_panel, 1)

    # ------------------------- PROCESAMIENTO -------------------------
    def extract_keypoints(self, results):
        """Extrae 126 keypoints de ambas manos (21 puntos × 3 coordenadas × 2 manos)."""
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

        return np.array(keypoints, dtype=np.float32)

    # ------------------------- FUNCIONES PRINCIPALES -------------------------
    def start_translation(self):
        os_name = platform.system().lower()
        if "windows" in os_name:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture("/dev/video0")

        if not self.cap.isOpened():
            self.current_translation_label.setText("❌ Error: No se puede acceder a la cámara")
            return

        self.sequence = []
        self.sentence = []
        self.translation_history = []
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.timer.start(30)

    def stop_translation(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.video_label.setText("Cámara detenida")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(frame_rgb)
        mp_drawing = mp.solutions.drawing_utils

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

        kp = self.extract_keypoints(results)
        self.sequence.append(kp)
        self.sequence = self.sequence[-10:]

        if len(self.sequence) == 10:
            try:
                res = self.model.predict(np.expand_dims(np.array(self.sequence), axis=0), verbose=0)[0]
                confidence = np.max(res)
                self.confidence_bar.setValue(int(confidence * 100))
                self.confidence_label.setText(f"Confianza: {confidence * 100:.1f}%")

                if confidence > self.threshold:
                    action = self.label_map[str(np.argmax(res))]
                    if not self.sentence or self.sentence[-1] != action:
                        self.sentence.append(action)
                        if len(self.sentence) > 5:
                            self.sentence = self.sentence[-5:]
                        self.current_translation_label.setText(f"✅ {action}")
                        self.translation_history.append(f"{action} ({confidence * 100:.1f}%)")
                        self.history_text.setPlainText("\n".join(reversed(self.translation_history)))

                        def speak():
                            engine = pyttsx3.init()
                            engine.say(action)
                            engine.runAndWait()
                        threading.Thread(target=speak, daemon=True).start()
            except Exception as e:
                print(f"Error en predicción: {e}")

        # Mostrar video
        h, w, ch = frame.shape
        qt_image = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.width(), self.video_label.height(),
                                                 Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self, event):
        self.stop_translation()
        event.accept()

# ------------------------- EJECUCIÓN -------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignTranslatorGUI()
    window.show()
    sys.exit(app.exec_())
