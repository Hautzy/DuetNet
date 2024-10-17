import tensorflow as tf

# Define your custom layer
class CustomLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(CustomLayer, self).__init__()

    def build(self, input_shape):
        pass

    def call(self, inputs):
        print(inputs.shape)  # This will now print the fixed shape
        return inputs * 2

# Define a function to build and save the model
def build_and_save_model():
    # Define the input shape with a fixed batch size of 32
    batch_size = 32
    input_shape = (batch_size, 256, 256, 3)  # Fixed batch size of 32, 256x256 image with 3 channels

    # Define the model with fixed batch size
    inputs = tf.keras.Input(batch_size=batch_size, shape=input_shape[1:])  # Include batch_size explicitly
    x = CustomLayer()(inputs)
    outputs = x
    model = tf.keras.Model(inputs, outputs)

    # Optionally print model summary to verify
    model.summary()

    # Save the model
    model.save('custom_model.h5')

# Call the function to build and save the model
build_and_save_model()
