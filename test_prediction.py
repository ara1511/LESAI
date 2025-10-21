"""
Script de prueba para verificar la compatibilidad del modelo
"""

import numpy as np
import tensorflow as tf
import json
import os

def test_model_prediction():
    print("🔄 Probando la predicción del modelo...")
    
    try:
        # Cargar modelo y label_map
        print("📂 Cargando modelo...")
        model = tf.keras.models.load_model('src/models/actions_smart.keras')
        
        with open('src/models/label_map_smart.json', 'r') as f:
            label_map = json.load(f)
        
        print(f"✅ Modelo cargado. Clases: {list(label_map.keys())}")
        print(f"📊 Forma de entrada esperada: {model.input_shape}")
        
        # Crear datos de prueba con la forma correcta
        # El modelo espera: (batch_size, 15, 63)
        test_sequence = np.random.random((1, 15, 63)).astype(np.float32)
        
        print(f"🧪 Datos de prueba: {test_sequence.shape}")
        
        # Hacer predicción
        prediction = model.predict(test_sequence, verbose=0)[0]
        print(f"🎯 Predicción exitosa: {prediction.shape}")
        print(f"📈 Valores: {prediction}")
        
        # Encontrar clase predicha
        max_idx = np.argmax(prediction)
        max_confidence = prediction[max_idx]
        
        predicted_action = None
        for action, idx in label_map.items():
            if idx == max_idx:
                predicted_action = action
                break
        
        print(f"🎉 Clase predicha: {predicted_action} (índice {max_idx})")
        print(f"📊 Confianza: {max_confidence:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_prediction()