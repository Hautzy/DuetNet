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

# Define the Input layer with flexible batch size but fixed feature size (256, 128)
input_tensor = Input(shape=(None, 256, 128))  # None for batch size

# Define the waveform generation logic
def generate_waveform_from_input(input_noise):
    fac = (args.seconds // 23) + 1
    return U.generate_waveform(input_noise, gen_ema, dec, dec2, batch_size=64)

# Apply the waveform generation function to the input tensor
waveform_output = generate_waveform_from_input(input_tensor)

# Create the Model
waveform_model = Model(inputs=input_tensor, outputs=waveform_output)

# Force the model to build with a fixed input shape (batch size = 1 for building)
print('Forcing model build with batch size 1...')
dummy_input = tf.random.normal(shape=(1, 256, 128))  # Fixed shape for building
waveform_model(dummy_input)  # This forces the internal graph to be built

# After building, the model can accept variable batch sizes during inference

# Ensure the export directory exists and save the model
if not os.path.isdir(export_folder):
    os.mkdir(export_folder)

# Save the model for later use, including TensorFlow.js conversion
waveform_model.save(f'./{export_folder}/waveform_model')
