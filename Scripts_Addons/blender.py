import numpy as np

print("1. Loading Night 1 and Night 2 datasets...")
# Load Night 1 data arrays
X1 = np.load('X_1.npy')
Y_CA_1 = np.load('Y_CA_1.npy')
Y_OSA_1 = np.load('Y_OSA_1.npy')
t1 = np.load('segment_times_n1.npy')

# Load Night 2 data arrays
X2 = np.load('X_2.npy')
Y_CA_2 = np.load('Y_CA_2.npy')
Y_OSA_2 = np.load('Y_OSA_2.npy')
t2 = np.load('segment_times_n2.npy')

print(f"   Night 1 Shape: X={X1.shape}, Y_CA={Y_CA_1.shape}, Y_OSA={Y_OSA_1.shape}")
print(f"   Night 2 Shape: X={X2.shape}, Y_CA={Y_CA_2.shape}, Y_OSA={Y_OSA_2.shape}")

print("2. Concatenating datasets...")
# axis=0 means we are stacking the segments on top of each other
X_combined = np.concatenate((X1, X2), axis=0)
Y_CA_combined = np.concatenate((Y_CA_1, Y_CA_2), axis=0)
Y_OSA_combined = np.concatenate((Y_OSA_1, Y_OSA_2), axis=0)
t_combined = np.concatenate((t1, t2), axis=0)

print("3. Saving the Master Dataset...")
# Saving with the exact filenames your training scripts expect!
np.save('X_train_PentaLSTM.npy', X_combined)
np.save('Y_train_Labels_CA.npy', Y_CA_combined)
np.save('Y_train_Labels_OSA.npy', Y_OSA_combined)
np.save('segment_times.npy', t_combined)

print(f"✅ SUCCESS! Master Dataset created.")
print(f"   Total Segments combined: {X_combined.shape[0]}")