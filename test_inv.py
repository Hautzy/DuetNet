import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input


class InverseSTFTLayer(Layer):
    def __init__(self, hop_length, frame_length, fft_length):
        super(InverseSTFTLayer, self).__init__()
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.fft_length = fft_length

    def call(self, inputs):
        S, P = inputs

        P_real = tf.cast(P, tf.float32)
        exp_real = tf.math.cos(P_real)
        exp_imag = tf.math.sin(P_real)

        S_real = tf.cast(S, tf.float32)
        S_imag = tf.zeros_like(S_real)

        SP_real = S_real * exp_real - S_imag * exp_imag
        SP_imag = S_real * exp_imag + S_imag * exp_real

        SP = tf.complex(SP_real, SP_imag)

        wv = tf.signal.inverse_stft(
            SP,
            frame_step=self.hop_length,
            frame_length=self.frame_length,
            fft_length=self.fft_length,
            window_fn=tf.signal.inverse_stft_window_fn(self.hop_length)
        )
        return wv


# Define the model
class CustomModel(Model):
    def __init__(self, hop_length, frame_length, fft_length):
        super(CustomModel, self).__init__()
        self.inverse_stft_layer = InverseSTFTLayer(hop_length, frame_length, fft_length)

    def call(self, inputs):
        return self.inverse_stft_layer(inputs)


# Instantiate the model
hop_length = 256
frame_length = 4 * hop_length
fft_length = 4 * hop_length

model = CustomModel(hop_length, frame_length, fft_length)

# Dummy inputs
S = tf.random.uniform([2, 100, 129], dtype=tf.float32)
P = tf.random.uniform([2, 100, 129], dtype=tf.float32)

# Call the model
output = model((S, P))

# Save the model
model.save('inverse_stft_model')

# Verify the shape of the output
print(output.shape)
