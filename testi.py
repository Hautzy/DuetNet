import tensorflow as tf

'''
hop = 256

frame_length = 4 * hop
frame_step = hop

window_fn = tf.signal.inverse_stft_window_fn(frame_step)

P = tf.random.normal(shape=(4096, 513))
S = tf.random.normal(shape=(4096, 513))

real_part = S * tf.cos(P)
imag_part = S * tf.sin(P)
SP = tf.complex(real_part, imag_part)
print('SP: ', SP.shape)
frames = tf.signal.irfft(SP, fft_length=[4 * hop])

window = window_fn(frame_length, dtype=tf.float32)
windowed_frames = frames * window

# Overlap-and-add to reconstruct the time-domain signal
reconstructed_signal = tf.signal.overlap_and_add(windowed_frames, frame_step)
print(reconstructed_signal.shape)
'''


import numpy as np
import tensorflow as tf

# Example data
chls_np = [np.array([1, 2, 3]), np.array([4, 5, 6])]
chls_tf = [tf.constant([1, 2, 3]), tf.constant([4, 5, 6])]

# NumPy stack
stacked_np = np.stack(chls_np, axis=-1)
print('NumPy stacked shape:', stacked_np.shape)  # Output: (3, 2)

# TensorFlow stack
stacked_tf = tf.stack(chls_tf, axis=-1)
print('TensorFlow stacked shape:', stacked_tf.shape)  # Output: (3, 2)
