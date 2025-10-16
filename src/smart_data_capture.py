"""
Solución práctica inmediata para mejorar la precisión del modelo
Sin dependencias problemáticas - usando solo lo que ya funciona
"""
import cv2
import numpy as np
import mediapipe as mp
import os
from pathlib import Path
import json
import tensorflow as tf
from datetime import datetime

class SmartDataCapture:
    """Capturador inteligente de datos con validación automática"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.8,  # Mayor precisión
            min_tracking_confidence=0.8
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Criterios de calidad más estrictos
        self.min_keypoints_ratio = 0.7  # 70% mínimo de keypoints
        self.min_movement_variance = 0.0001  # Movimiento mínimo
        self.stable_frames_needed = 15  # Frames estables antes de capturar
        
    def calculate_quality_metrics(self, keypoints_sequence):
        """Calcular múltiples métricas de calidad"""
        if len(keypoints_sequence) == 0:
            return 0.0, {}
        
        sequence_array = np.array(keypoints_sequence)
        metrics = {}
        
        # 1. Ratio de keypoints detectados
        non_zero_ratio = np.count_nonzero(sequence_array) / sequence_array.size
        metrics['keypoints_ratio'] = non_zero_ratio
        
        # 2. Consistencia entre frames
        frame_differences = []
        for i in range(1, len(sequence_array)):
            diff = np.mean(np.abs(sequence_array[i] - sequence_array[i-1]))
            frame_differences.append(diff)
        
        metrics['frame_consistency'] = 1.0 - np.std(frame_differences) if frame_differences else 0
        
        # 3. Variabilidad de movimiento (debe haber algo de movimiento)
        movement_variance = np.var(sequence_array[sequence_array != 0])
        metrics['movement_variance'] = min(movement_variance * 1000, 1.0)
        
        # 4. Distribución espacial (manos no deben estar en una esquina)
        non_zero_points = sequence_array[sequence_array != 0]
        if len(non_zero_points) > 0:
            spatial_spread = np.std(non_zero_points)
            metrics['spatial_distribution'] = min(spatial_spread * 10, 1.0)
        else:
            metrics['spatial_distribution'] = 0.0
        
        # Score final ponderado
        final_score = (
            metrics['keypoints_ratio'] * 0.4 +
            metrics['frame_consistency'] * 0.2 +
            metrics['movement_variance'] * 0.2 +
            metrics['spatial_distribution'] * 0.2
        )
        
        return final_score, metrics
    
    def extract_keypoints(self, results):
        """Extraer keypoints con validación"""
        keypoints = np.zeros(126)
        
        if results.multi_hand_landmarks:
            hands_count = min(len(results.multi_hand_landmarks), 2)
            
            for hand_idx in range(hands_count):
                hand_landmarks = results.multi_hand_landmarks[hand_idx]
                start_idx = hand_idx * 63
                
                for i, landmark in enumerate(hand_landmarks.landmark):
                    if i < 21:  # Solo 21 landmarks por mano
                        idx = start_idx + (i * 3)
                        keypoints[idx:idx+3] = [landmark.x, landmark.y, landmark.z]
        
        return keypoints
    
    def capture_high_quality_data(self, word, target_samples=30):
        """Capturar datos de alta calidad con validación automática"""
        
        print(f"\n🎯 CAPTURA INTELIGENTE PARA: '{word.upper()}'")
        print("=" * 50)
        print("📋 INSTRUCCIONES MEJORADAS:")
        print("  • Haz la seña de forma CLARA y CONSISTENTE")
        print("  • Mantén las manos dentro del frame")
        print("  • Iluminación uniforme (evita sombras)")
        print("  • Repite la seña varias veces de forma natural")
        print("  • El sistema capturará automáticamente cuando detecte calidad alta")
        print("  • Presiona 'q' para terminar, ESC para cancelar")
        
        # Configurar captura
        data_dir = Path(f"data/keypoints_smart/{word}")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Mayor resolución
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Variables de estado
        sequence_buffer = []
        captured_samples = 0
        stable_frames = 0
        last_capture_time = 0
        
        # Estadísticas
        total_frames = 0
        quality_history = []
        
        print(f"\n🚀 Iniciando captura... (Objetivo: {target_samples} muestras)")
        
        while captured_samples < target_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            total_frames += 1
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Extraer keypoints
            keypoints = self.extract_keypoints(results)
            sequence_buffer.append(keypoints)
            
            # Mantener buffer de 15 frames
            if len(sequence_buffer) > 15:
                sequence_buffer.pop(0)
            
            # Calcular calidad si tenemos suficientes frames
            quality_score = 0
            metrics = {}
            
            if len(sequence_buffer) >= 10:
                quality_score, metrics = self.calculate_quality_metrics(sequence_buffer)
                quality_history.append(quality_score)
            
            # Auto-captura si calidad es consistentemente alta
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            
            if (quality_score > 0.7 and 
                len(sequence_buffer) == 15 and 
                current_time - last_capture_time > 2.0):  # Mínimo 2 segundos entre capturas
                
                # Verificar que la calidad es estable
                recent_quality = quality_history[-5:] if len(quality_history) >= 5 else quality_history
                if len(recent_quality) >= 3 and min(recent_quality) > 0.6:
                    
                    # Guardar muestra de alta calidad
                    sequence_array = np.array(sequence_buffer)
                    
                    # Generar múltiples variaciones para data augmentation
                    for variation in range(3):  # 3 variaciones por captura
                        augmented_sequence = self.apply_augmentation(sequence_array, variation)
                        
                        filename = data_dir / f"smart_{captured_samples:03d}_{variation}.npy"
                        np.save(filename, augmented_sequence)
                    
                    captured_samples += 1
                    last_capture_time = current_time
                    
                    print(f"✅ AUTO-CAPTURA #{captured_samples} - Calidad: {quality_score:.3f}")
                    
                    # Limpiar buffer para próxima captura
                    sequence_buffer = []
            
            # Dibujar interfaz
            self.draw_capture_interface(frame, results, word, captured_samples, 
                                      target_samples, quality_score, metrics)
            
            # Mostrar frame
            cv2.imshow(f"Captura Inteligente: {word}", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 27:  # ESC
                print("\n❌ Captura cancelada")
                cap.release()
                cv2.destroyAllWindows()
                return False
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Estadísticas finales
        avg_quality = np.mean(quality_history) if quality_history else 0
        
        print(f"\n✅ CAPTURA COMPLETADA PARA '{word}':")
        print(f"  • Muestras capturadas: {captured_samples} (x3 variaciones = {captured_samples * 3} archivos)")
        print(f"  • Calidad promedio: {avg_quality:.3f}")
        print(f"  • Frames procesados: {total_frames}")
        print(f"  • Tasa de éxito: {(captured_samples / (total_frames / 100)):.1f} capturas por 100 frames")
        
        return True
    
    def apply_augmentation(self, sequence, variation_type):
        """Aplicar aumentación de datos sutil pero efectiva"""
        augmented = sequence.copy().astype(np.float32)
        
        if variation_type == 0:
            # Original con ruido mínimo
            noise = np.random.normal(0, 0.001, augmented.shape)
            augmented += noise
            
        elif variation_type == 1:
            # Escalado ligero
            scale = np.random.uniform(0.98, 1.02)
            augmented *= scale
            
        elif variation_type == 2:
            # Desplazamiento temporal suave
            shift = np.random.randint(1, 3)
            augmented = np.roll(augmented, shift, axis=0)
        
        return augmented
    
    def draw_capture_interface(self, frame, results, word, captured, target, quality, metrics):
        """Dibujar interfaz de captura mejorada"""
        
        # Dibujar landmarks si existen
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
        
        # Panel de información
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Información principal
        cv2.putText(frame, f"Palabra: {word.upper()}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        
        cv2.putText(frame, f"Progreso: {captured}/{target}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Barra de progreso
        progress_width = int(300 * captured / target)
        cv2.rectangle(frame, (20, 80), (320, 100), (100, 100, 100), -1)
        cv2.rectangle(frame, (20, 80), (20 + progress_width, 100), (0, 255, 0), -1)
        
        # Calidad
        quality_color = (0, 255, 0) if quality > 0.7 else (0, 165, 255) if quality > 0.5 else (0, 0, 255)
        cv2.putText(frame, f"Calidad: {quality:.3f}", (20, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, quality_color, 2)
        
        # Métricas detalladas
        if metrics:
            y_pos = 150
            for metric, value in metrics.items():
                if metric == 'keypoints_ratio':
                    text = f"Deteccion: {value:.2f}"
                elif metric == 'movement_variance':
                    text = f"Movimiento: {value:.2f}"
                else:
                    continue
                    
                cv2.putText(frame, text, (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_pos += 20
        
        # Instrucciones
        cv2.putText(frame, "Auto-captura activada | Q=Salir | ESC=Cancelar", 
                   (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

def main():
    """Función principal para capturar datos inteligentemente"""
    print("🚀 SISTEMA DE CAPTURA INTELIGENTE DE DATOS")
    print("=" * 50)
    
    capturer = SmartDataCapture()
    words = ["hola", "gracias", "si", "no"]
    
    print("\n💡 VENTAJAS DE ESTE SISTEMA:")
    print("• Auto-captura cuando detecta alta calidad")
    print("• Métricas de calidad en tiempo real")
    print("• Aumentación automática de datos")
    print("• Mayor resolución de cámara")
    print("• Validación inteligente")
    
    for i, word in enumerate(words):
        print(f"\n📋 Palabra {i+1}/{len(words)}: '{word}'")
        response = input(f"¿Capturar datos mejorados para '{word}'? (s/n): ")
        
        if response.lower().startswith('s'):
            success = capturer.capture_high_quality_data(word, target_samples=25)
            if not success:
                print("❌ Captura interrumpida")
                break
        else:
            print(f"⏭️ Saltando '{word}'")
    
    print("\n🎯 SIGUIENTE PASO: Entrenar modelo con datos mejorados")
    print("Ejecuta: python src/train_with_smart_data.py")

if __name__ == "__main__":
    main()