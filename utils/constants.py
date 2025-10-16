# CONFIGURACIÓN GLOBAL

SEQUENCE_LENGTH = 10          # Número de frames por secuencia
KEYPOINT_DIM = 126            # 21 puntos x 3 coordenadas x 2 manos = 126
NUM_SEQUENCES = 15            # Número de secuencias por acción

# 4 palabras básicas para entrenar
ACTIONS = ['hola', 'gracias', 'si', 'no']

# Estas 4 palabras son perfectas para empezar:
# - Muy fáciles de encontrar en YouTube
# - Movimientos claros y diferenciados  
# - Útiles en cualquier conversación

# Rutas
DATA_PATH = 'data/frame_actions'
KEYPOINTS_PATH = 'data/keypoints'
MODELS_PATH = 'models'
