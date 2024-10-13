import os
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from parse.parse_generate import parse_args
from models import Models_functions
from utils import Utils_functions
import numpy as np
from scipy.io.wavfile import write as write_wav

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

# Define the Input layer with a fixed input shape, avoiding None entirely
# Here, you set a fixed batch size (e.g., 1 or another value based on your scenario)
input_tensor = Input(shape=(256, 128))  # No None value in the input shape

# Define the waveform generation logic that requires fixed input sizes
def generate_waveform_from_input(input_noise):
    if input_noise.shape[0] is None:
        print('Using dummy data to build the model...')
        fac = (args.seconds // 23) + 1
        input_noise = U.get_noise_interp_multi(fac, args.truncation)
    return U.generate_waveform(input_noise, gen_ema, dec, dec2, batch_size=64)

# Apply the waveform generation function to the input tensor
waveform_output = generate_waveform_from_input(input_tensor)

# Create the model
waveform_model = Model(inputs=input_tensor, outputs=waveform_output)

# Generate a fixed-size dummy input and build the model
print('Building the model with fixed input data...')
dummy_input = tf.random.normal(shape=(6, 256, 128))  # Use the fixed batch size
waveform_model(dummy_input)  # Build the model with the fixed input

# Ensure the export directory exists
if not os.path.isdir(export_folder):
    os.mkdir(export_folder)

# Save the model for later use (e.g., TensorFlow.js conversion)
waveform_model.save(f'./{export_folder}/waveform_model')

# For inference, handle batch sizes manually:
# When running inference, always ensure the input size is fixed by generating real input data
def run_inference(model, batch_size):
    # Generate real input noise with a fixed shape to pass to the model
    input_noise = U.get_noise_interp_multi(batch_size, args.truncation)
    return model(input_noise)

# Example inference with a batch size of 10
batch_size = 10
inference_result = run_inference(waveform_model, batch_size)
write_wav(f"out.wav", args.sr, np.squeeze(inference_result)[: args.seconds * args.sr])
print('Inference complete.')
