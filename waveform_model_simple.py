import os
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from parse.parse_generate import parse_args
from models import Models_functions
from utils import Utils_functions

# Define export folder
export_folder = 'exported_models'

# Parse arguments and initialize utility and model functions
args = parse_args()
U = Utils_functions(args)
M = Models_functions(args)
M.download_networks()

# Retrieve the GAN-related models
models_ls = M.get_networks()
critic, gen, enc, dec, enc2, dec2, gen_ema, [opt_dec, opt_disc], switch = models_ls

# Force the batch size to be fixed during model building
fixed_batch_size = 1  # Use a small fixed batch size to build the model
input_tensor = Input(shape=(fixed_batch_size, 256, 128))  # No None, force fixed size

# Define the waveform generation logic
def generate_waveform_from_input(input_noise):
    return U.generate_waveform(input_noise, gen_ema, dec, dec2, batch_size=64)

# Apply the waveform generation function to the input tensor
waveform_output = generate_waveform_from_input(input_tensor)

# Create the Model with a fixed batch size
waveform_model = Model(inputs=input_tensor, outputs=waveform_output)

# Generate the fixed dummy input and build the model
print('Forcing model build with a fixed batch size of 1...')
dummy_input = tf.random.normal(shape=(6, 256, 128))  # Fixed shape for building
waveform_model(dummy_input)  # This forces the internal graph to be built

# Now you can redefine the model to accept variable batch sizes (inference mode)
# Create a new model with a flexible input batch size
input_tensor_flexible = Input(shape=(None, 256, 128))  # Allow None for flexible batch size
waveform_output_flexible = generate_waveform_from_input(input_tensor_flexible)

# Create the final model with a flexible batch size for inference
waveform_model_flexible = Model(inputs=input_tensor_flexible, outputs=waveform_output_flexible)

# Ensure the export directory exists and save the final flexible model
if not os.path.isdir(export_folder):
    os.mkdir(export_folder)

# Save the flexible model for later use, including TensorFlow.js conversion
waveform_model_flexible.save(f'./{export_folder}/waveform_model_flexible')
