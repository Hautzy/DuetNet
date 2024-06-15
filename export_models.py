import os
from parse.parse_generate import parse_args
from models import Models_functions

export_folder = 'exported_models'

args = parse_args()

M = Models_functions(args)
M.download_networks()
models_ls = M.get_networks()
critic, gen, enc, dec, enc2, dec2, gen_ema, [opt_dec, opt_disc], switch = models_ls

# check if folder exists if not create one
if not os.path.isdir(export_folder):
    os.mkdir(export_folder)

dec.save(f'./{export_folder}/dec_model')
dec2.save(f'./{export_folder}/dec2_model')
gen_ema.save(f'./{export_folder}/gen_ema_model')