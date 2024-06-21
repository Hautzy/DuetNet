import tensorflow as tf
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.models import Model

# Define a custom layer that applies IRFFT
class IRFFTLayer(Layer):
    def __init__(self):
        super(IRFFTLayer, self).__init__()

    def manual_inverse_stft(self, complex_spectrogram, frame_length, frame_step):
        # Use the complementary window function
        # window_fn = tf.signal.inverse_stft_window_fn(frame_step)

        # Perform the inverse real FFT
        real_frames = tf.signal.irfft(complex_spectrogram, fft_length=[frame_length])

        '''
        # Apply the window function
        window = window_fn(frame_length, dtype=tf.float32)
        windowed_frames = real_frames * window

        # Overlap-and-add to reconstruct the time-domain signal
        reconstructed_signal = tf.signal.overlap_and_add(windowed_frames, frame_step)

        return reconstructed_signal
        '''
        return real_frames

    def call(self, inputs):
        hop = 256
        S = tf.random.normal(shape=(513, 2))
        P = tf.random.normal(shape=(513, 2))

        real_part = S * tf.cos(P)  # Real component
        imag_part = S * tf.sin(P)  # Imaginary component
        complex_tensor = tf.complex(real_part, imag_part)

        return tf.signal.irfft(complex_tensor)


input_tensor = Input(shape=(1,))
custom_layer = IRFFTLayer()(input_tensor)  # Output is the result of the IRFFT

model = Model(inputs=input_tensor, outputs=custom_layer)

model.build(input_shape=(None, 1))

model.save('./saved_model_irfft_only')

print("Model saved")