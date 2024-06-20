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

        # Initialize empty lists to collect real and imaginary parts of frames
        frames_real = []
        frames_imag = []

        for i in range(SP_real.shape[1]):
            real_part = SP_real[:, i, :]
            imag_part = SP_imag[:, i, :]

            # Perform inverse FFT on real and imaginary parts separately
            time_frame_real = tf.signal.irfft(real_part, fft_length=self.fft_length)
            time_frame_imag = tf.signal.irfft(imag_part, fft_length=self.fft_length)

            frames_real.append(time_frame_real)
            frames_imag.append(time_frame_imag)

        frames_real = tf.stack(frames_real, axis=1)
        frames_imag = tf.stack(frames_imag, axis=1)

        # Overlap and add frames to reconstruct the signal
        num_frames = tf.shape(frames_real)[1]
        frame_length = tf.shape(frames_real)[2]
        signal_length = num_frames * self.hop_length + frame_length - self.hop_length
        signal_real = tf.zeros([tf.shape(frames_real)[0], signal_length])
        signal_imag = tf.zeros([tf.shape(frames_imag)[0], signal_length])

        for i in range(num_frames):
            start = i * self.hop_length
            signal_real[:, start:start + frame_length] += frames_real[:, i, :]
            signal_imag[:, start:start + frame_length] += frames_imag[:, i, :]

        # Combine the real and imaginary parts at the end
        signal = tf.sqrt(signal_real ** 2 + signal_imag ** 2)
        return signal


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
