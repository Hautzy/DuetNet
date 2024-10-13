import os
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.models import Model
from parse.parse_generate import parse_args
from models import Models_functions
from utils import Utils_functions


class GenerateWaveformLayer(Layer):
    def __init__(self, gen_ema, dec, dec2, **kwargs):
        super(GenerateWaveformLayer, self).__init__(**kwargs)
        self.gen_ema = gen_ema
        self.dec = dec
        self.dec2 = dec2

    def call(self, inputs):
        return U.generate_waveform(inputs, self.gen_ema, self.dec, self.dec2, batch_size=64)


export_folder = 'exported_models'

args = parse_args()

U = Utils_functions(args)

M = Models_functions(args)
M.download_networks()
models_ls = M.get_networks()
critic, gen, enc, dec, enc2, dec2, gen_ema, [opt_dec, opt_disc], switch = models_ls

input_tensor = Input(shape=(256, 128))
waveform_layer = GenerateWaveformLayer(gen_ema, dec, dec2)(input_tensor)

waveform_model = Model(inputs=input_tensor, outputs=waveform_layer)

waveform_model.build(input_shape=(None, 256, 128))

if not os.path.isdir(export_folder):
    os.mkdir(export_folder)

waveform_model.save(f'./{export_folder}/waveform_model')
