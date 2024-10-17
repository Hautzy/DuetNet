from parse.parse_generate import parse_args
from utils import Utils_functions
import tensorflow as tf

args = parse_args()

U = Utils_functions(args)

def continuous_noise_interp(noiseg, right_noisel, fac=1, var=2.0):
    coordratio = args.coordlen // args.latlen

    noisels = []
    if right_noisel is None:
        noisels.append(
            tf.concat([U.truncated_normal([1, 64], var, dtype=tf.float32), noiseg], -1)
        )
    else:
        noisels.append(right_noisel)

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

fac = 1
var = 2.0

noiseg = U.truncated_normal([1, args.coorddepth], var, dtype=tf.float32)
right_noisel = None

noise, right_noisel = continuous_noise_interp(noiseg, right_noisel, fac, var)
print(noise.shape)
print(right_noisel.shape)
noise, right_noisel = continuous_noise_interp(noiseg, right_noisel, fac, var)
print(noise.shape)
print(right_noisel.shape)
