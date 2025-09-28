# CONFIGURACIÓN GLOBAL

SEQUENCE_LENGTH = 10          # Número de frames por secuencia (menos para probar rápido)
KEYPOINT_DIM = 126            # 21 puntos x 3 coordenadas x 2 manos = 126

# Solo 2 palabras para empezar
ACTIONS = ['hola', 'adios']

# Rutas
DATA_PATH = 'data/frame_actions'
KEYPOINTS_PATH = 'data/keypoints'
MODELS_PATH = 'models'