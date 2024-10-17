import os
from parse.parse_generate import parse_args
from utils import Utils_functions
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.models import Model

export_folder = 'exported_models'

args = parse_args()

U = Utils_functions(args)

def continuous_noise_interp(noiseg, right_noisel, fac=1, var=2.0):
    coordratio = args.coordlen // args.latlen

    noisels = [right_noisel]
    for k in range(3 + ((fac - 1) // coordratio) - 1):
        noisels.append(tf.concat([U.truncated_normal([1, 64], var, dtype=tf.float32), noiseg], -1))

    rls = tf.concat(
        [
            tf.linspace(noisels[k], noisels[k + 1], args.coordlen + 1, axis=-2)[:, :-1, :]
            for k in range(len(noisels) - 1)
        ],
        -2,
    )

    rls = U.center_coordinate(rls)
    rls = rls[:, args.latlen // 4:, :]
    rls = rls[:, : (rls.shape[-2] // args.latlen) * args.latlen, :]
    rls = tf.split(rls, rls.shape[-2] // args.latlen, -2)
    return tf.concat(rls[:fac], 0), noisels[-1]

class ContinuousNoiseLayer(Layer):
    def __init__(self, **kwargs):
        super(ContinuousNoiseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs):
        noiseg, right_noisel = inputs
        print('noiseg ', noiseg.shape)
        print('right_noisel ', right_noisel.shape)
        return continuous_noise_interp(noiseg, right_noisel)

def build_model():

    noiseg_input = Input(batch_size=1, shape=(64))
    right_noisel_input = Input(batch_size=1, shape=(128))

    continuous_noise_layer = ContinuousNoiseLayer()([noiseg_input, right_noisel_input])
    continous_noise_model = Model(inputs=[noiseg_input, right_noisel_input], outputs=continuous_noise_layer)

    var = 2.0
    noiseg = U.truncated_normal([1, args.coorddepth], var, dtype=tf.float32)
    right_noisel = tf.concat([U.truncated_normal([1, 64], var, dtype=tf.float32), noiseg], -1)

    print('Using actual data to build the model...')

    for k in range(10):
        noise, right_noisel = continous_noise_model([noiseg, right_noisel])
        print(noise.shape)
        print(right_noisel.shape)

    if not os.path.isdir(export_folder):
        os.mkdir(export_folder)

    continous_noise_model.save(f'./{export_folder}/continous_noise_model')

def test():
    fac = 1
    var = 2.0
    noiseg = U.truncated_normal([1, args.coorddepth], var, dtype=tf.float32)
    right_noisel = tf.concat([U.truncated_normal([1, 64], var, dtype=tf.float32), noiseg], -1)

    print(noiseg.shape)

    noise, right_noisel = continuous_noise_interp(noiseg, right_noisel, fac, var)
    print(noise.shape)
    print(right_noisel.shape)
    noise, right_noisel = continuous_noise_interp(noiseg, right_noisel, fac, var)
    print(noise.shape)
    print(right_noisel.shape)


build_model()
#test()