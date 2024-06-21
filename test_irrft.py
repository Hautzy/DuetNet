import tensorflow as tf

# Define a custom layer that applies IRFFT
class IRFFTLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(IRFFTLayer, self).__init__()

    def call(self, inputs):
        if inputs.shape[0] is None:
            inputs = tf.random.normal(shape=(513, 2), dtype=tf.complex64)
        # Apply IRFFT. The input is expected to be a complex tensor suitable for IRFFT
        return tf.signal.irfft(inputs)

# Create a model that uses the IRFFT layer
# Input shape is designed for IRFFT input requirements
inputs = tf.keras.Input(shape=(513, 2), dtype=tf.complex64)
custom_layer = IRFFTLayer()(inputs)  # Output is the result of the IRFFT

model = tf.keras.Model(inputs=inputs, outputs=custom_layer)

# Since the output of IRFFT is real, there's no need to compile the model if it's only for inference
# However, if you want to train this model, you should compile it with a real-valued loss function
# Example: model.compile(optimizer='adam', loss='mean_squared_error')

model.build(input_shape=(513, 2))

# Save the model
model_path = 'saved_model_irfft_only'
model.save(model_path)

print("Model saved to:", model_path)