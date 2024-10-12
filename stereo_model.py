import os
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.models import Model
from parse.parse_generate import parse_args
from models import Models_functions
from utils import Utils_functions


class GenerateStereoLayer(Layer):
    def __init__(self, model_ls, **kwargs):
        super(GenerateStereoLayer, self).__init__(**kwargs)
        self.model_ls = model_ls

    def call(self, inputs):
        return U.generate_example_stereo(self.model_ls)


export_folder = 'exported_models'

args = parse_args()

U = Utils_functions(args)

M = Models_functions(args)
M.download_networks()
models_ls = M.get_networks()

input_tensor = Input(shape=(1,))  # Dummy shape
stereo_layer = GenerateStereoLayer(models_ls)(input_tensor)

stereo_model = Model(inputs=input_tensor, outputs=stereo_layer)

stereo_model.build(input_shape=(None, 1))  # Dummy shape

if not os.path.isdir(export_folder):
    os.mkdir(export_folder)

stereo_model.save(f'./{export_folder}/stereo_model')
