import tensorflow as tf
from scipy.io.wavfile import write as write_wav
import numpy as np
from parse.parse_generate import parse_args
from utils import Utils_functions
import json
import matplotlib.pyplot as plt

args = parse_args()

U = Utils_functions(args)

# read json file and convert into 2d tensor
# Load the tensor data from the JSON file
with open('tensor.json', 'r') as file:
    tensor_data = json.load(file)

# Convert the list of lists into a TensorFlow tensor
tensor = tf.constant(tensor_data)

# Verify the tensor
print(f'Tensor shape: {tensor.shape}')
print(f'Tensor dtype: {tensor.dtype}')
print(tensor)

write_wav(f"out.wav", 44100, np.squeeze(tensor))

# Prepare the data for plotting
data = np.flip(
    np.array(
        tf.transpose(
            U.wv2spec_hop((tensor[:, 0] + tensor[:, 1]) / 2.0, 80.0, args.hop * 2),
            [1, 0],
        )
    ),
    -2,
)

# Create a single plot
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(data, cmap=None)
ax.axis("off")
ax.set_title("Generated1")
plt.show()