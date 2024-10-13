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

# Define the Input layer without explicitly specifying batch size,
# allowing it to be inferred dynamically during inference
input_tensor = Input(shape=(256, 128))  # No batch size in the shape, allowing flexibility

# Define the waveform generation logic
def generate_waveform_from_input(input_noise):
    fac = (args.seconds // 23) + 1
    return U.generate_waveform(input_noise, gen_ema, dec, dec2, batch_size=64)

# Apply the waveform generation function to the input tensor
waveform_output = generate_waveform_from_input(input_tensor)

# Create the model
waveform_model = Model(inputs=input_tensor, outputs=waveform_output)

# Generate a fixed-size dummy input and build the model
print('Building the model with real input data...')
dummy_input = tf.random.normal(shape=(1, 256, 128))  # Use a fixed batch size for building
waveform_model(dummy_input)  # Build the model with the fixed input

# Ensure the export directory exists
if not os.path.isdir(export_folder):
    os.mkdir(export_folder)

# Save the model for later use (e.g., TensorFlow.js conversion)
waveform_model.save(f'./{export_folder}/waveform_model')
