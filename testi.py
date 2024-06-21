import tensorflow as tf

hop = 256

S = tf.random.normal(shape=(513, 2))
P = tf.random.normal(shape=(513, 2))
SP = tf.cast(S, tf.complex64) * tf.math.exp(1j * tf.cast(P, tf.complex64))
wv = tf.signal.inverse_stft(
    SP,
    4 * hop,
    hop,
    fft_length=4 * hop,
    window_fn=tf.signal.inverse_stft_window_fn(hop),
)
print(wv.shape)