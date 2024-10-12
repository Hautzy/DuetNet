import os

from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.models import Model

from parse.parse_generate import parse_args
from utils import Utils_functions


class NoiseLayer(Layer):
    def __init__(self, fac, trunc, **kwargs):
        super(NoiseLayer, self).__init__(**kwargs)
        self.fac = fac
        self.trunc = trunc

    def call(self, inputs):
        res = U.get_noise_interp_multi(self.fac, self.trunc)
        print('noise', res.shape)
        return res


export_folder = 'exported_models'

args = parse_args()

U = Utils_functions(args)


input_tensor = Input(shape=(1,))
noise_layer = NoiseLayer(6, 2.0)(input_tensor)

noise_model = Model(inputs=input_tensor, outputs=noise_layer)

noise_model.build(input_shape=(None, 1))  # Dummy shape

if not os.path.isdir(export_folder):
    os.mkdir(export_folder)

noise_model.save(f'./{export_folder}/noise_model')
