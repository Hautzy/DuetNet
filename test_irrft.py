import tensorflow as tf
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.models import Model

# Define a custom layer that applies IRFFT
class IRFFTLayer(Layer):
    def __init__(self):
        super(IRFFTLayer, self).__init__()

    def call(self, inputs):
        if inputs.shape[0] is None:
            real_tensor = tf.random.normal(shape=(513, 2))
            imag_tensor = tf.random.normal(shape=(513, 2))

            inputs = tf.complex(real_tensor, imag_tensor)
        # Apply IRFFT. The input is expected to be a complex tensor suitable for IRFFT
        return tf.signal.irfft(inputs)


# Combine them into a complex tensor
input_tensor = Input(shape=(2))
custom_layer = IRFFTLayer()(input_tensor)  # Output is the result of the IRFFT

model = tf.keras.Model(inputs=input_tensor, outputs=custom_layer)

real_tensor = tf.random.normal(shape=(513, 2))
imag_tensor = tf.random.normal(shape=(513, 2))

input = tf.complex(real_tensor, imag_tensor)

res = model(input)

print(res.shape)

model.save('saved_model_irfft_only')

print("Model saved")