import sys
import os
import platform
import threading  # Para que la voz no bloquee la GUI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import os
os.environ["QT_QPA_PLATFORM"] = "windows"
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json
import pyttsx3
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QVBoxLayout, QWidget, QGridLayout, QHBoxLayout,
                            QFrame, QGroupBox, QProgressBar, QTextEdit, QSplitter)
from PyQt5.QtCore import QTimer, Qt, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette, QIcon
from utils.constants import MODELS_PATH

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
                text-decoration: none;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
                min-width: 120px;
                min-height: 40px;
            }}
            QPushButton:hover {{
                background-color: {self.lighten_color(self.base_color)};
                transform: translateY(-2px);
            }}
            QPushButton:pressed {{
                background-color: {self.darken_color(self.base_color)};
                transform: translateY(0px);
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

class SignTranslatorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🖐️🤖 Comunicación Inclusiva con IA")
        self.setWindowIcon(QIcon('assets/icon.png'))
        self.setGeometry(100, 100, 1400, 900)
        
        # Aplicar tema oscuro moderno
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 2px solid #3c3c3c;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
                background-color: #2d2d2d;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLabel {
                color: #ffffff;
            }
            QTextEdit {
                background-color: #2d2d2d;
                border: 2px solid #3c3c3c;
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
                color: #ffffff;
            }
            QProgressBar {
                border: 2px solid #3c3c3c;
                border-radius: 8px;
                text-align: center;
                background-color: #2d2d2d;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 6px;
            }
        """)

        # Inicializar MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

        # Cargar modelo
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

        # Timer para actualizar video
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def setup_ui(self):
        # Widget central y layout principal
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setSpacing(20)
        self.main_layout.setContentsMargins(20, 20, 20, 20)

        # Panel izquierdo - Video y controles
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setSpacing(15)

        # Grupo de video
        self.video_group = QGroupBox("📹 Cámara en Tiempo Real")
        self.video_layout = QVBoxLayout(self.video_group)
        
        # Frame del video con borde
        self.video_frame = QFrame()
        self.video_frame.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.video_frame.setLineWidth(2)
        self.video_frame.setStyleSheet("""
            QFrame {
                border: 3px solid #4CAF50;
                border-radius: 12px;
                background-color: #000000;
            }
        """)
        self.video_frame_layout = QVBoxLayout(self.video_frame)
        
        self.video_label = QLabel("Cámara sin inicializar")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        self.video_frame_layout.addWidget(self.video_label)
        
        self.video_layout.addWidget(self.video_frame)

        # Panel de controles
        self.controls_group = QGroupBox("🎮 Controles")
        self.controls_layout = QHBoxLayout(self.controls_group)
        self.controls_layout.setSpacing(15)

        self.start_btn = AnimatedButton("▶️ Iniciar Traducción", "#4CAF50")
        self.start_btn.clicked.connect(self.start_translation)
        
        self.stop_btn = AnimatedButton("⏹️ Detener", "#f44336")
        self.stop_btn.clicked.connect(self.stop_translation)
        self.stop_btn.setEnabled(False)

        self.controls_layout.addWidget(self.start_btn)
        self.controls_layout.addWidget(self.stop_btn)
        self.controls_layout.addStretch()

        self.left_layout.addWidget(self.video_group)
        self.left_layout.addWidget(self.controls_group)

        # Panel derecho - Traducción e información
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setSpacing(15)

        # Grupo de traducción actual
        self.translation_group = QGroupBox("🔤 Traducción Actual")
        self.translation_layout = QVBoxLayout(self.translation_group)
        
        self.current_translation_label = QLabel("Esperando señas...")
        self.current_translation_label.setAlignment(Qt.AlignCenter)
        self.current_translation_label.setStyleSheet("""
            QLabel {
                background-color: #2d2d2d;
                border: 2px solid #4CAF50;
                border-radius: 12px;
                padding: 20px;
                font-size: 28px;
                font-weight: bold;
                color: #4CAF50;
                min-height: 80px;
            }
        """)
        self.translation_layout.addWidget(self.current_translation_label)

        # Barra de confianza
        self.confidence_label = QLabel("Confianza: 0%")
        self.confidence_label.setStyleSheet("font-size: 14px; color: #cccccc;")
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        
        self.translation_layout.addWidget(self.confidence_label)
        self.translation_layout.addWidget(self.confidence_bar)

        # Grupo de historial
        self.history_group = QGroupBox("📝 Historial de Traducciones")
        self.history_layout = QVBoxLayout(self.history_group)
        
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        self.history_text.setMaximumHeight(200)
        self.history_text.setPlaceholderText("Las traducciones aparecerán aquí...")
        self.history_layout.addWidget(self.history_text)

        # Grupo de estadísticas
        self.stats_group = QGroupBox("📊 Estadísticas")
        self.stats_layout = QVBoxLayout(self.stats_group)
        
        self.words_translated_label = QLabel("Palabras traducidas: 0")
        self.session_time_label = QLabel("Tiempo de sesión: 00:00")
        self.avg_confidence_label = QLabel("Confianza promedio: 0%")
        
        for label in [self.words_translated_label, self.session_time_label, self.avg_confidence_label]:
            label.setStyleSheet("font-size: 14px; padding: 5px; color: #cccccc;")
            self.stats_layout.addWidget(label)

        # Agregar grupos al panel derecho
        self.right_layout.addWidget(self.translation_group)
        self.right_layout.addWidget(self.history_group)
        self.right_layout.addWidget(self.stats_group)
        self.right_layout.addStretch()

        # Agregar paneles al layout principal
        self.main_layout.addWidget(self.left_panel, 2)  # 2/3 del espacio
        self.main_layout.addWidget(self.right_panel, 1)  # 1/3 del espacio

    def extract_keypoints(self, results):
        keypoints = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    keypoints.extend([landmark.x, landmark.y, landmark.z])
        while len(keypoints) < 126:
            keypoints.append(0)
        return np.array(keypoints[:126])

    def start_translation(self):
        os_name = platform.system().lower()
        if "windows" in os_name:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture("/dev/video0")

        if not self.cap.isOpened():
            self.current_translation_label.setText("❌ Error: No se puede acceder a la cámara")
            self.current_translation_label.setStyleSheet("""
                QLabel {
                    background-color: #2d2d2d;
                    border: 2px solid #f44336;
                    border-radius: 12px;
                    padding: 20px;
                    font-size: 28px;
                    font-weight: bold;
                    color: #f44336;
                    min-height: 80px;
                }
            """)
            return

        self.sequence = []
        self.sentence = []
        self.translation_history = []
        self.words_count = 0

        # UI update
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.current_translation_label.setText("🔍 Analizando señas...")
        self.current_translation_label.setStyleSheet("""
            QLabel {
                background-color: #2d2d2d;
                border: 2px solid #FF9800;
                border-radius: 12px;
                padding: 20px;
                font-size: 28px;
                font-weight: bold;
                color: #FF9800;
                min-height: 80px;
            }
        """)

        self.timer.start(30)  # Actualizar cada 30ms

    def stop_translation(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
            
        # Actualizar UI
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.video_label.setText("Cámara detenida")
        self.video_label.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        self.current_translation_label.setText("⏹️ Traducción detenida")
        self.current_translation_label.setStyleSheet("""
            QLabel {
                background-color: #2d2d2d;
                border: 2px solid #888888;
                border-radius: 12px;
                padding: 20px;
                font-size: 28px;
                font-weight: bold;
                color: #888888;
                min-height: 80px;
            }
        """)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        # Dibujar landmarks con mejor visualización
        if results.multi_hand_landmarks:
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # Extraer keypoints
        kp = self.extract_keypoints(results)
        self.sequence.append(kp)
        self.sequence = self.sequence[-10:]  # Mantener últimos 10 frames

        if len(self.sequence) == 10:
            res = self.model.predict(np.expand_dims(np.array(self.sequence), axis=0), verbose=0)[0]
            self.current_confidence = res[np.argmax(res)]
            
            # Actualizar barra de confianza
            confidence_percent = int(self.current_confidence * 100)
            self.confidence_bar.setValue(confidence_percent)
            self.confidence_label.setText(f"Confianza: {confidence_percent}%")
            
            if self.current_confidence > self.threshold:
                action_idx = np.argmax(res)
                action = self.label_map[str(action_idx)]
                
                if not self.sentence or self.sentence[-1] != action:
                    self.sentence.append(action)
                    if len(self.sentence) > 5:
                        self.sentence = self.sentence[-5:]
                    
                    # Actualizar traducción actual
                    current_text = " ".join(self.sentence)
                    self.current_translation_label.setText(f"✅ {current_text}")
                    self.current_translation_label.setStyleSheet("""
                        QLabel {
                            background-color: #2d2d2d;
                            border: 2px solid #4CAF50;
                            border-radius: 12px;
                            padding: 20px;
                            font-size: 28px;
                            font-weight: bold;
                            color: #4CAF50;
                            min-height: 80px;
                        }
                    """)
                    
                    # Agregar al historial
                    self.translation_history.append(f"• {action} (Confianza: {confidence_percent}%)")
                    if len(self.translation_history) > 20:
                        self.translation_history = self.translation_history[-20:]
                    
                    self.history_text.setPlainText("\n".join(reversed(self.translation_history)))
                    
                    # Actualizar estadísticas
                    self.words_count += 1
                    self.words_translated_label.setText(f"Palabras traducidas: {self.words_count}")
                    
                    # Calcular confianza promedio
                    confidences = [float(line.split('(Confianza: ')[1].split('%')[0]) 
                                 for line in self.translation_history if 'Confianza:' in line]
                    if confidences:
                        avg_conf = sum(confidences) / len(confidences)
                        self.avg_confidence_label.setText(f"Confianza promedio: {avg_conf:.1f}%")
                    
                    # Hablar en un hilo separado para no bloquear la UI
                    def speak_text():
                        try:
                            engine = pyttsx3.init()
                            engine.say(action)
                            engine.runAndWait()
                        except Exception as e:
                            print(f"Error en TTS: {e}")
                    
                    tts_thread = threading.Thread(target=speak_text)
                    tts_thread.daemon = True
                    tts_thread.start()

        # Mostrar frame en QLabel con mejor escalado
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # Escalar manteniendo aspecto y ajustando al contenedor
        scaled_pixmap = pixmap.scaled(
            self.video_label.width(), 
            self.video_label.height(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        self.stop_translation()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignTranslatorGUI()
    window.show()
    sys.exit(app.exec_())