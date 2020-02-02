__all__ = ['SEQ_LEN', 'LSTM_UNITS', 'FEATURE_COLUMNS', 'TARGET_COLUMNS', 'BATCH_SIZE']

SEQ_LEN = 64
LSTM_UNITS = 64
FEATURE_COLUMNS = ['x', 'y', 'z', 'Vx', 'Vy', 'Vz']
TARGET_COLUMNS = ['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']
BATCH_SIZE = 16