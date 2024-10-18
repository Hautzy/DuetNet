import os
from parse.parse_generate import parse_args
from utils import Utils_functions
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from continuous_noise_model import continuous_noise_interp
from models import Models_functions
from utils import Utils_functions


args = parse_args()

U = Utils_functions(args)

M = Models_functions(args)
M.download_networks()
models_ls = M.get_networks()
critic, gen, enc, dec, enc2, dec2, gen_ema, [opt_dec, opt_disc], switch = models_ls

fac = 1
var = 2.0
noiseg = U.truncated_normal([1, args.coorddepth], var, dtype=tf.float32)
right_noisel = tf.concat([U.truncated_normal([1, 64], var, dtype=tf.float32), noiseg], -1)


noise, right_noisel = continuous_noise_interp(noiseg, right_noisel, fac, var)
wv0 = U.generate_waveform(noise, gen_ema, dec, dec2, batch_size=64)

noise, right_noisel = continuous_noise_interp(noiseg, right_noisel, fac, var)
wv1 = U.generate_waveform(noise, gen_ema, dec, dec2, batch_size=64)


noise, right_noisel = continuous_noise_interp(noiseg, right_noisel, fac, var)
wv2 = U.generate_waveform(noise, gen_ema, dec, dec2, batch_size=64)

# concat wv tensors
result = tf.concat([wv0, wv1, wv2], axis=0)

print(result.shape)

# Convert TensorFlow tensor to numpy for plotting
waveform = result.numpy()

# Plotting the waveform
plt.figure(figsize=(10, 6))

# Plotting the first channel
plt.plot(waveform[:, 0], label="Channel 1")

# Plotting the second channel
plt.plot(waveform[:, 1], label="Channel 2")

# Add vertical lines at the boundaries where tensors were concatenated
plt.axvline(x=wv0.shape[0], color='red', linestyle='--', label='Connection Point 1')
plt.axvline(x=2*wv0.shape[0], color='green', linestyle='--', label='Connection Point 2')

# Adding labels and legend
plt.title('Waveforms with Connection Points')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()

output_image_path = 'waveform_plot.png'
plt.savefig(output_image_path)

# Close the plot to free up memory
plt.close()