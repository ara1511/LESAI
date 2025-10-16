#!/usr/bin/env python3
"""
Script de captura de datos completamente automatizado
Captura las 5 señas específicas: hola, gracias, si, no, buenos_dias
"""

import cv2
import numpy as np
import mediapipe as mp
import os
from pathlib import Path
import time

def create_sample_data():
    """Crear datos de muestra para las 5 señas"""
    
    signs = ["hola", "gracias", "si", "no", "buenos_dias"]
    
    print("🚀 CREANDO ESTRUCTURA DE DATOS")
    print("=" * 50)
    
    # Crear directorios
    for sign in signs:
        output_dir = f"data/keypoints_smart/{sign}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print(f"📁 Creado directorio para '{sign}'")
    
    # Configurar MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    print("\n🎥 Iniciando captura automática...")
    print("💡 Haz las señas cuando se te indique")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se puede acceder a la cámara")
        return False
    
    for sign_index, sign in enumerate(signs):
        print(f"\n🎯 Capturando '{sign.upper()}' ({sign_index + 1}/{len(signs)})")
        
        samples_captured = 0
        target_samples = 15  # Menos muestras para ir más rápido
        
        # Mostrar mensaje por 3 segundos
        start_time = time.time()
        while time.time() - start_time < 3:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"Preparate para: {sign.upper()}", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f"Comenzando en {3 - int(time.time() - start_time)}...", 
                           (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('LESAI - Captura Automatica', frame)
                cv2.waitKey(30)
        
        # Capturar muestras
        consecutive_detections = 0
        while samples_captured < target_samples:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                consecutive_detections += 1
                
                # Si detectamos manos por suficiente tiempo, capturar
                if consecutive_detections >= 30:  # 1 segundo a 30fps
                    # Capturar secuencia de 15 frames
                    sequence = []
                    for _ in range(15):
                        ret, frame = cap.read()
                        if ret:
                            frame = cv2.flip(frame, 1)
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = hands.process(rgb_frame)
                            
                            if results.multi_hand_landmarks:
                                hand_landmarks = results.multi_hand_landmarks[0]
                                keypoints = []
                                for landmark in hand_landmarks.landmark:
                                    keypoints.extend([landmark.x, landmark.y, landmark.z])
                                sequence.append(keypoints)
                            
                            # Mostrar progreso
                            cv2.putText(frame, f"Capturando {sign.upper()}: {len(sequence)}/15", 
                                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv2.putText(frame, f"Muestra: {samples_captured + 1}/{target_samples}", 
                                       (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                            cv2.imshow('LESAI - Captura Automatica', frame)
                            cv2.waitKey(30)
                    
                    # Guardar si tenemos secuencia completa
                    if len(sequence) == 15:
                        samples_captured += 1
                        filename = f"data/keypoints_smart/{sign}/{samples_captured}.npy"
                        np.save(filename, np.array(sequence))
                        print(f"  ✅ Muestra {samples_captured}/{target_samples} de '{sign}' guardada")
                        consecutive_detections = 0
                        time.sleep(0.5)  # Pausa entre capturas
            else:
                consecutive_detections = 0
            
            # Mostrar estado
            cv2.putText(frame, f"Seña: {sign.upper()}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Muestras: {samples_captured}/{target_samples}", 
                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            if results.multi_hand_landmarks:
                cv2.putText(frame, "✅ Manos detectadas", (50, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "❌ Muestra las manos", (50, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('LESAI - Captura Automatica', frame)
            
            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        print(f"  🎯 Completado '{sign}': {samples_captured} muestras")
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    return True

def main():
    print("🚀 LESAI - CAPTURA AUTOMÁTICA DE SEÑAS")
    print("=" * 60)
    print("📋 Señas a capturar: hola, gracias, si, no, buenos_dias")
    print("🎯 15 muestras por seña = 75 muestras totales")
    
    success = create_sample_data()
    
    if success:
        print(f"\n✅ CAPTURA COMPLETADA!")
        
        # Verificar datos creados
        total_samples = 0
        for sign in ["hola", "gracias", "si", "no", "buenos_dias"]:
            sign_dir = f"data/keypoints_smart/{sign}"
            if os.path.exists(sign_dir):
                samples = len([f for f in os.listdir(sign_dir) if f.endswith('.npy')])
                total_samples += samples
                print(f"   📂 {sign}: {samples} muestras")
        
        print(f"\n📊 Total: {total_samples} muestras")
        
        if total_samples >= 30:  # Al menos 2 señas con datos
            print(f"\n🚀 LISTO PARA ENTRENAR!")
            print(f"   Ejecuta: python src/train_final.py")
        else:
            print(f"\n⚠️  Se necesitan más muestras para entrenar")
    else:
        print(f"\n❌ Error en la captura")

if __name__ == "__main__":
    main()