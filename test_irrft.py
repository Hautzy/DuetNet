import tensorflow as tf
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.models import Model

# Define a custom layer that applies IRFFT
class IRFFTLayer(Layer):
    def __init__(self):
        super(IRFFTLayer, self).__init__()

    def call(self, inputs):
        real_tensor = tf.random.normal(shape=(513, 2))
        imag_tensor = tf.random.normal(shape=(513, 2))

        inputs = tf.complex(real_tensor, imag_tensor)
        return tf.signal.irfft(inputs)


input_tensor = Input(shape=(1,))
custom_layer = IRFFTLayer()(input_tensor)  # Output is the result of the IRFFT

model = tf.keras.Model(inputs=input_tensor, outputs=custom_layer)

model.build(input_shape=(None, 1))

model.save('./saved_model_irfft_only')

print("Model saved")