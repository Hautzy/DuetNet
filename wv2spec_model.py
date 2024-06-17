import os
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.models import Model
from parse.parse_generate import parse_args
from models import Models_functions
from utils import Utils_functions


class Wv2SpecLayer(Layer):
    def __init__(self, topdb, hopsize, **kwargs):
        super(Wv2SpecLayer, self).__init__(**kwargs)
        self.topdb = topdb
        self.hopsize = hopsize
    def call(self, inputs):
        if inputs.shape[0] is None:
            inputs = tf.random.normal([128, 2])
        res = U.wv2spec_hop((inputs[:, 0] + inputs[:, 1]) / 2.0, self.topdb, self.hopsize * 2)
        res = tf.transpose(res, [1, 0])
        print('Result')
        print(res.shape)
        return res


export_folder = 'exported_models'

args = parse_args()

U = Utils_functions(args)

input_tensor = Input(shape=(2))
wv2spec_layer = Wv2SpecLayer(80.0, args.hop)(input_tensor)

wv2spec_model = Model(inputs=input_tensor, outputs=wv2spec_layer)

dummy_input = tf.random.normal([256, 2])
res = wv2spec_model(dummy_input)
print(res.shape)

if not os.path.isdir(export_folder):
    os.mkdir(export_folder)

wv2spec_model.save(f'./{export_folder}/wv2spec_model_model')
