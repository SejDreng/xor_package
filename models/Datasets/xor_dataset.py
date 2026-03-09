import numpy as np

# Create 100 samples of XOR-style inputs (0.0 or 1.0)
# Shape should match your model input, e.g., (100, 2)
data = np.random.choice([0.0, 1.0], size=(100, 2)).astype(np.float32)
np.save("xor_calibration_data.npy", data)