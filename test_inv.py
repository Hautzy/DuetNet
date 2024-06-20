import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.models import Model


class CustomInverseSTFT(Layer):
    def __init__(self, frame_length, frame_step, fft_length, **kwargs):
        super(CustomInverseSTFT, self).__init__(**kwargs)
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length

    def call(self, inputs):
        S_real, S_imag = inputs
        return self.custom_inverse_stft(S_real, S_imag)

    def custom_inverse_stft(self, S_real, S_imag):
        """
        Custom inverse STFT function that handles real and imaginary parts separately.
        Args:
        - S_real (Tensor): Real part of the STFT result.
        - S_imag (Tensor): Imaginary part of the STFT result.

        Returns:
        - Tensor: Reconstructed time-domain signal.
        """

        def manual_ifft(real, imag, fft_length):
            """
            Perform an inverse FFT manually on the real and imaginary parts.
            """
            n = tf.cast(tf.shape(real)[-1], tf.float32)
            k = tf.range(0, fft_length, dtype=tf.float32)
            k = tf.reshape(k, (1, -1))
            exp_term_real = tf.cos(2.0 * tf.constant(np.pi) * k / n)
            exp_term_imag = tf.sin(2.0 * tf.constant(np.pi) * k / n)

            real_ifft = tf.matmul(real, exp_term_real) - tf.matmul(imag, exp_term_imag)
            imag_ifft = tf.matmul(real, exp_term_imag) + tf.matmul(imag, exp_term_real)

            return real_ifft / n, imag_ifft / n

        # Number of frames and length of each frame
        num_frames = tf.shape(S_real)[0]

        # Hann window function
        window = tf.signal.hann_window(self.frame_length, periodic=True)

        # Initialize the output signal array
        output_length = self.frame_step * (num_frames - 1) + self.frame_length
        output_signal = tf.zeros([output_length], dtype=tf.float32)
        window_correction = tf.zeros([output_length], dtype=tf.float32)

        # Perform the inverse FFT manually and overlap-add
        for i in range(num_frames):
            # Reconstruct the time-domain frame using manual IFFT
            real_ifft, imag_ifft = manual_ifft(S_real[i], S_imag[i], self.fft_length)

            # Combine real and imaginary parts (imag part should be zero ideally)
            time_frame = real_ifft - imag_ifft

            # Apply window function
            time_frame = time_frame * window

            # Overlap-add the frame into the output signal
            start = i * self.frame_step
            for j in range(self.frame_length):
                output_signal = tf.tensor_scatter_nd_add(output_signal, [[start + j]], [time_frame[j]])
                window_correction = tf.tensor_scatter_nd_add(window_correction, [[start + j]], [window[j]])

        # Correct for the windowing
        output_signal = output_signal / tf.maximum(window_correction, 1e-8)

        return output_signal


# Example usage
S = tf.random.uniform([100, 256], dtype=tf.float32)  # Magnitude spectrogram
P = tf.random.uniform([100, 256], dtype=tf.float32) * 2 * tf.constant(np.pi)  # Phase spectrogram

# Separate the real and imaginary parts
S_real = S * tf.math.cos(P)
S_imag = S * tf.math.sin(P)

frame_length = 256
frame_step = 64
fft_length = 256

# Define the model with the custom layer
inputs_real = Input(shape=(None, fft_length))
inputs_imag = Input(shape=(None, fft_length))
outputs = CustomInverseSTFT(frame_length, frame_step, fft_length)([inputs_real, inputs_imag])
model = Model(inputs=[inputs_real, inputs_imag], outputs=outputs)

# Perform the custom inverse STFT using the model
custom_wv = model([S_real, S_imag])
