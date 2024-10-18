import os
from parse.parse_generate import parse_args
from utils import Utils_functions
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
result = tf.concat([wv0, wv1, wv2])

print(result.shape)