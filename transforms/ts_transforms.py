import random
import numpy as np

# Custom data augmentation for time series with multiple features
def augment_time_series(data):
    shift = np.random.randint(1, data.shape[1])
    shifted_data = np.roll(data, shift, axis=1)
    noise = np.random.normal(0, 0.01, data.shape)
    noisy_data = data + noise
    scale = np.random.uniform(0.8, 1.2)
    scaled_data = data * scale
    permuted_data = np.random.permutation(data.T).T
    augmented_data = random.choice([shifted_data, noisy_data, scaled_data, permuted_data])
    return augmented_data

# Subsampling function for low resolution
def subsample(data, factor=2):
    return data[:, ::factor]