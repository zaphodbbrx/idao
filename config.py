__all__ = ['SEQ_LEN', 'LSTM_UNITS', 'FEATURE_COLUMNS', 'TARGET_COLUMNS', 'BATCH_SIZE']

SEQ_LEN = 25
LSTM_UNITS = 8
TARGET_COLUMNS = ['x', 'y', 'z', 'Vx', 'Vy', 'Vz']
# FEATURE_COLUMNS = ['x', 'y', 'z']
FEATURE_COLUMNS = ['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']
# TARGET_COLUMNS = ['x_sim', 'y_sim', 'z_sim']
BATCH_SIZE = 64