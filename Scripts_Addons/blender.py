import numpy as np

print("1. Loading Night 1 and Night 2 datasets...")
# Load Night 1 (Mostly CA)
X1 = np.load('X_train_n1.npy')
Y1 = np.load('Y_train_n1.npy')

# Load Night 2 (Mostly OSA)
X2 = np.load('X_train_n2.npy')
Y2 = np.load('Y_train_n2.npy')

print(f"   Night 1 Shape: X={X1.shape}, Y={Y1.shape}")
print(f"   Night 2 Shape: X={X2.shape}, Y={Y2.shape}")

# Load nights segments
t1 = np.load('segment_times_n1.npy')
t2 = np.load('segment_times_n2.npy')

print("2. Concatenating datasets...")
# axis=0 means we are stacking the segments on top of each other
X_combined = np.concatenate((X1, X2), axis=0)
Y_combined = np.concatenate((Y1, Y2), axis=0)
t_combined = np.concatenate((t1, t2), axis=0)

print("3. Saving the Master Dataset...")
np.save('X_train_Master.npy', X_combined)
np.save('Y_train_Master.npy', Y_combined)
np.save('segment_times_Master.npy', t_combined)


print(f"✅ SUCCESS! Master Dataset created.")
print(f"   Total Segments: {X_combined.shape[0]}")