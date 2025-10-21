# CONFIGURACIÓN GLOBAL

SEQUENCE_LENGTH = 10          # Número de frames por secuencia
KEYPOINT_DIM = 216            # 21 puntos x 3 coordenadas x 2 manos = 126

# Palabras a reconocer
ACTIONS = ['gracias', 'hola', 'adios']

# Número de secuencias (muestras) a capturar por acción
NO_SEQUENCES = 30

# Rutas
DATA_PATH = 'data/frame_actions'    # Ya no se usará
KEYPOINTS_PATH = 'data/keypoints'
MODELS_PATH = 'src/models'