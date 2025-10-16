"""
GUI Bonita con Lógica Simple que SÍ Funciona
"""
import sys
import os
import platform
import cv2
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
            }}
            QPushButton:pressed {{
                background-color: {self.darken_color(self.base_color)};
            }}
        """
    
    def lighten_color(self, color):
        return "#5CBF60" if color == "#4CAF50" else "#FF6B6B"
    
    def darken_color(self, color):
        return "#45A049" if color == "#4CAF50" else "#FF5252"

class SignTranslatorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🤟 Traductor de Lengua de Señas - Versión Mejorada")
        self.setGeometry(100, 100, 1400, 900)
        
        # Variables principales
        self.cap = None
        self.model = None
        self.label_map = {}
        self.sequence = []
        self.current_word = ""
        self.sentence = []
        self.is_detecting = False
        
        # Estadísticas
        self.words_count = 0
        self.translation_history = []
        
        # Configurar MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Timer para video
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        
        # Configurar interfaz y cargar modelo
        self.setup_ui()
        self.load_model()
        self.start_camera()
        
    def setup_ui(self):
        """Configurar interfaz bonita"""
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
        
        # Título
        title_label = QLabel("📹 Cámara en Tiempo Real")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2C3E50; margin: 10px;")
        self.left_layout.addWidget(title_label)
        
        # Video
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("""
            QLabel {
                border: 3px solid #3498DB;
                border-radius: 15px;
                background-color: #F8F9FA;
            }
        """)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("🎥 Iniciando cámara...")
        self.left_layout.addWidget(self.video_label)
        
        # Controles
        controls_group = QGroupBox("🎮 Controles")
        controls_group.setFont(QFont("Arial", 12, QFont.Bold))
        controls_layout = QHBoxLayout(controls_group)
        
        self.btn_toggle = AnimatedButton("▶️ INICIAR DETECCIÓN", "#4CAF50")
        self.btn_toggle.clicked.connect(self.toggle_detection)
        controls_layout.addWidget(self.btn_toggle)
        
        self.btn_clear = AnimatedButton("🗑️ LIMPIAR", "#FF5722")
        self.btn_clear.clicked.connect(self.clear_translation)
        controls_layout.addWidget(self.btn_clear)
        
        self.left_layout.addWidget(controls_group)
        
        # Panel derecho - Resultados y estadísticas
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setSpacing(15)
        
        # Traducción actual
        translation_group = QGroupBox("🎯 Traducción Actual")
        translation_group.setFont(QFont("Arial", 12, QFont.Bold))
        translation_layout = QVBoxLayout(translation_group)
        
        self.current_translation_label = QLabel("🤲 Presiona INICIAR y muestra las manos")
        self.current_translation_label.setAlignment(Qt.AlignCenter)
        self.current_translation_label.setStyleSheet("""
            QLabel {
                background-color: #E8F4FD;
                border: 2px solid #3498DB;
                border-radius: 12px;
                padding: 20px;
                font-size: 24px;
                font-weight: bold;
                color: #2C3E50;
                min-height: 80px;
            }
        """)
        translation_layout.addWidget(self.current_translation_label)
        
        # Barra de confianza
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Confianza:"))
        
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #BDC3C7;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #27AE60;
                border-radius: 6px;
            }
        """)
        confidence_layout.addWidget(self.confidence_bar)
        
        self.confidence_label = QLabel("0%")
        self.confidence_label.setMinimumWidth(60)
        confidence_layout.addWidget(self.confidence_label)
        
        translation_layout.addLayout(confidence_layout)
        self.right_layout.addWidget(translation_group)
        
        # Estado del sistema
        status_group = QGroupBox("📊 Estado del Sistema")
        status_group.setFont(QFont("Arial", 12, QFont.Bold))
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("⏸️ Detección pausada")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #FFF3E0;
                border: 2px solid #FF9800;
                border-radius: 8px;
                padding: 10px;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        status_layout.addWidget(self.status_label)
        self.right_layout.addWidget(status_group)
        
        # Historial
        history_group = QGroupBox("📝 Historial de Traducciones")
        history_group.setFont(QFont("Arial", 12, QFont.Bold))
        history_layout = QVBoxLayout(history_group)
        
        self.history_text = QTextEdit()
        self.history_text.setMaximumHeight(200)
        self.history_text.setStyleSheet("""
            QTextEdit {
                background-color: #F8F9FA;
                border: 2px solid #DEE2E6;
                border-radius: 8px;
                padding: 10px;
                font-family: 'Courier New';
                font-size: 12px;
            }
        """)
        history_layout.addWidget(self.history_text)
        self.right_layout.addWidget(history_group)
        
        # Estadísticas
        stats_group = QGroupBox("📈 Estadísticas")
        stats_group.setFont(QFont("Arial", 12, QFont.Bold))
        stats_layout = QVBoxLayout(stats_group)
        
        self.words_translated_label = QLabel("Palabras traducidas: 0")
        self.avg_confidence_label = QLabel("Confianza promedio: 0%")
        
        for label in [self.words_translated_label, self.avg_confidence_label]:
            label.setStyleSheet("font-size: 14px; padding: 5px;")
            stats_layout.addWidget(label)
        
        self.right_layout.addWidget(stats_group)
        
        # Agregar paneles al layout principal
        self.main_layout.addWidget(self.left_panel, 2)
        self.main_layout.addWidget(self.right_panel, 1)
        
        # Estilo general de la ventana
        self.setStyleSheet("""
            QMainWindow {
                background-color: #FFFFFF;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #BDC3C7;
                border-radius: 10px;
                margin: 5px;
                padding-top: 15px;
                background-color: #FAFAFA;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
                color: #2C3E50;
            }
        """)
    
    def load_model(self):
        """Cargar modelo y configuraciones"""
        try:
            print("🔄 Cargando modelo...")
            self.model = tf.keras.models.load_model('models/actions_smart.keras')
            with open('models/label_map_smart.json', 'r') as f:
                self.label_map = json.load(f)
            print("✅ Modelo cargado exitosamente")
            return True
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            self.current_translation_label.setText("❌ Error: No se puede cargar el modelo")
            return False
    
    def start_camera(self):
        """Iniciar cámara"""
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.timer.start(30)  # 30ms = ~33 FPS
            
    def toggle_detection(self):
        """Activar/desactivar detección"""
        self.is_detecting = not self.is_detecting
        
        if self.is_detecting:
            self.btn_toggle.setText("⏸️ PAUSAR DETECCIÓN")
            self.btn_toggle.base_color = "#FF5722"
            self.btn_toggle.setStyleSheet(self.btn_toggle.get_base_style())
            self.status_label.setText("🟢 Detección ACTIVA - Haz una seña clara")
            self.status_label.setStyleSheet("""
                QLabel {
                    background-color: #E8F5E8;
                    border: 2px solid #4CAF50;
                    border-radius: 8px;
                    padding: 10px;
                    font-size: 16px;
                    font-weight: bold;
                    color: #2E7D32;
                }
            """)
        else:
            self.btn_toggle.setText("▶️ INICIAR DETECCIÓN")
            self.btn_toggle.base_color = "#4CAF50"
            self.btn_toggle.setStyleSheet(self.btn_toggle.get_base_style())
            self.status_label.setText("⏸️ Detección PAUSADA")
            self.status_label.setStyleSheet("""
                QLabel {
                    background-color: #FFF3E0;
                    border: 2px solid #FF9800;
                    border-radius: 8px;
                    padding: 10px;
                    font-size: 16px;
                    font-weight: bold;
                }
            """)
            self.sequence = []
            self.confidence_bar.setValue(0)
            self.confidence_label.setText("0%")
    
    def clear_translation(self):
        """Limpiar traducciones"""
        self.sentence = []
        self.current_word = ""
        self.translation_history = []
        self.words_count = 0
        self.sequence = []
        
        self.current_translation_label.setText("🧹 Historial limpiado")
        self.history_text.clear()
        self.words_translated_label.setText("Palabras traducidas: 0")
        self.avg_confidence_label.setText("Confianza promedio: 0%")
        self.confidence_bar.setValue(0)
        self.confidence_label.setText("0%")
    
    def extract_keypoints(self, results):
        """Extraer keypoints de las manos"""
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([lh, rh])
    
    def process_frame(self):
        """Procesar cada frame del video"""
        ret, frame = self.cap.read()
        if not ret:
            return
            
        # Voltear para efecto espejo
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar con MediaPipe
        results = self.holistic.process(rgb_frame)
        
        # Dibujar landmarks
        if results.left_hand_landmarks or results.right_hand_landmarks:
            mp_draw = mp.solutions.drawing_utils
            if results.left_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        
        # Solo detectar si está activo
        if self.is_detecting:
            self.detect_sign(results)
        
        # Mostrar frame
        self.display_frame(frame)
    
    def detect_sign(self, results):
        """LÓGICA SIMPLE QUE FUNCIONA"""
        # Solo procesar si hay manos
        if not (results.left_hand_landmarks or results.right_hand_landmarks):
            self.current_translation_label.setText("🤲 Muestra las manos frente a la cámara")
            self.sequence = []
            self.confidence_bar.setValue(0)
            self.confidence_label.setText("Sin manos")
            return
        
        # Extraer keypoints
        keypoints = self.extract_keypoints(results)
        self.sequence.append(keypoints)
        
        # Mantener solo últimos 15 frames
        if len(self.sequence) > 15:
            self.sequence = self.sequence[-15:]
        
        # Solo predecir cuando tengamos 15 frames completos
        if len(self.sequence) == 15:
            sequence_array = np.array(self.sequence)
            
            # Verificar que hay datos suficientes
            non_zero_ratio = np.count_nonzero(sequence_array) / sequence_array.size
            
            if non_zero_ratio > 0.4:  # Al menos 40% de datos
                # Normalización simple
                for frame in sequence_array:
                    non_zero_mask = frame != 0
                    if np.any(non_zero_mask):
                        mean_val = np.mean(frame[non_zero_mask])
                        std_val = np.std(frame[non_zero_mask])
                        if std_val > 0:
                            frame[non_zero_mask] = (frame[non_zero_mask] - mean_val) / std_val
                
                # Predicción
                try:
                    prediction = self.model.predict(np.expand_dims(sequence_array, axis=0), verbose=0)[0]
                    
                    # Criterios simples que funcionan
                    max_idx = np.argmax(prediction)
                    max_confidence = prediction[max_idx]
                    
                    # Segunda mayor confianza
                    sorted_indices = np.argsort(prediction)[::-1]
                    second_confidence = prediction[sorted_indices[1]]
                    difference = max_confidence - second_confidence
                    
                    # Actualizar barra de confianza
                    confidence_percent = int(max_confidence * 100)
                    self.confidence_bar.setValue(confidence_percent)
                    self.confidence_label.setText(f"{confidence_percent}%")
                    
                    # CRITERIOS SIMPLES: confianza > 60% Y diferencia > 30%
                    if max_confidence > 0.6 and difference > 0.3:
                        # Encontrar nombre de la acción
                        predicted_action = None
                        for action, idx in self.label_map.items():
                            if idx == max_idx:
                                predicted_action = action
                                break
                        
                        if predicted_action and predicted_action != self.current_word:
                            self.current_word = predicted_action
                            self.sentence.append(predicted_action)
                            
                            # Actualizar interfaz
                            current_text = " ".join(self.sentence[-5:])  # Últimas 5 palabras
                            self.current_translation_label.setText(f"🎯 {current_text.upper()}")
                            self.current_translation_label.setStyleSheet("""
                                QLabel {
                                    background-color: #E8F5E8;
                                    border: 2px solid #4CAF50;
                                    border-radius: 12px;
                                    padding: 20px;
                                    font-size: 28px;
                                    font-weight: bold;
                                    color: #2E7D32;
                                    min-height: 80px;
                                }
                            """)
                            
                            # Actualizar historial
                            self.translation_history.append(f"• {predicted_action} (Confianza: {confidence_percent}%)")
                            self.history_text.setPlainText("\n".join(reversed(self.translation_history[-10:])))
                            
                            # Estadísticas
                            self.words_count += 1
                            self.words_translated_label.setText(f"Palabras traducidas: {self.words_count}")
                            
                            # Hablar
                            self.speak(predicted_action)
                            
                            # Limpiar secuencia para próxima detección
                            self.sequence = []
                    else:
                        # Señal no clara
                        self.current_translation_label.setText(f"🤔 Señal no clara (Conf: {confidence_percent}%, Dif: {int(difference*100)}%)")
                        self.current_translation_label.setStyleSheet("""
                            QLabel {
                                background-color: #FFF9C4;
                                border: 2px solid #FFC107;
                                border-radius: 12px;
                                padding: 20px;
                                font-size: 18px;
                                font-weight: bold;
                                color: #F57F17;
                                min-height: 80px;
                            }
                        """)
                        
                except Exception as e:
                    self.current_translation_label.setText(f"❌ Error en predicción: {str(e)[:50]}")
            else:
                self.current_translation_label.setText("📊 Datos insuficientes - Haz una seña más clara")
    
    def speak(self, text):
        """Reproducir texto con voz"""
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except:
            print(f"No se pudo reproducir: {text}")
    
    def display_frame(self, frame):
        """Mostrar frame en la interfaz"""
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        qt_image = qt_image.rgbSwapped()  # BGR a RGB
        
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
    
    def closeEvent(self, event):
        """Limpiar al cerrar"""
        if self.cap:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reducir warnings de TensorFlow
    app = QApplication(sys.argv)
    window = SignTranslatorGUI()
    window.show()
    sys.exit(app.exec_())