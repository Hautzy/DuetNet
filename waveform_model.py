import os
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.models import Model
from parse.parse_generate import parse_args
from models import Models_functions
from utils import Utils_functions


export_folder = 'exported_models'

args = parse_args()

U = Utils_functions(args)

M = Models_functions(args)
M.download_networks()
models_ls = M.get_networks()
critic, gen, enc, dec, enc2, dec2, gen_ema, [opt_dec, opt_disc], switch = models_ls

class GenerateWaveformLayer(Layer):
    def __init__(self, gen_ema, dec, dec2, **kwargs):
        super(GenerateWaveformLayer, self).__init__(**kwargs)
        self.gen_ema = gen_ema
        self.dec = dec
        self.dec2 = dec2

    @tf.function
    def call(self, inputs):
        if inputs.shape[0] is None:
            print('Using dummy data to build the model...')
            fac = (args.seconds // 23) + 1
            inputs = U.get_noise_interp_multi(fac, args.truncation)
        return U.generate_waveform(inputs, self.gen_ema, self.dec, self.dec2, batch_size=64)

input_tensor = Input(shape=(256, 128))
waveform_layer = GenerateWaveformLayer(gen_ema, dec, dec2)(input_tensor)

waveform_model = Model(inputs=input_tensor, outputs=waveform_layer)

fac = (args.seconds // 23) + 1

print('Using actual data to build the model...')
dummy_input = U.get_noise_interp_multi(fac, args.truncation)
waveform_model(dummy_input)

if not os.path.isdir(export_folder):
    os.mkdir(export_folder)

waveform_model.save(f'./{export_folder}/waveform_model')
