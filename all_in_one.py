import tensorflow as tf
from parse.parse_test import parse_args
from models import Models_functions
from utils import Utils_functions
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model


class GenerateWaveformLayer(Layer):
    def __init__(self, gen_ema, dec, dec2, batch_size=64, **kwargs):
        super(GenerateWaveformLayer, self).__init__(**kwargs)
        self.gen_ema = gen_ema
        self.dec = dec
        self.dec2 = dec2
        self.batch_size = batch_size

    def call(self, inputs):
        return U.generate_waveform(inputs, self.gen_ema, self.dec, self.dec2, batch_size=self.batch_size)




if __name__ == "__main__":
    # parse args
    args = parse_args()

    # test musika
    U = Utils_functions(args)

    # Assuming `parse_args` and `Models_functions` are defined elsewhere
    args = parse_args()
    M = Models_functions(args)
    models_ls = M.get_networks()

    # Create an input layer
    #inpf = tf.keras.layers.Input((M.args.latlen, M.args.latdepth * 2))

    # Use the custom GenerateWaveformLayer to wrap the generate_waveform function
    #waveform_layer = GenerateWaveformLayer(gen_ema, dec, dec2, batch_size=64)(inpf)

    # Define the model
    #allinone = Model(inpf, [waveform_layer])

    # Save the model
    #allinone.save('./exported_models/allinone')