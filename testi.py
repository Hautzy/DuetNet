import tensorflow as tf

hop = 256

frame_length = 4 * hop
frame_step = hop

window_fn = tf.signal.inverse_stft_window_fn(frame_step)

P = tf.random.normal(shape=(4096, 513))
S = tf.random.normal(shape=(4096, 513))

'''
SP = tf.cast(S, tf.complex64) * tf.math.exp(1j * tf.cast(P, tf.complex64))
print('SP: ', SP.shape)
wv = tf.signal.inverse_stft(
    SP,
    4 * hop,
    hop,
    fft_length=4 * hop,
    window_fn=tf.signal.inverse_stft_window_fn(hop),
)
print('wv: ', wv.shape)
'''

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
