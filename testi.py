import tensorflow as tf

# Define tensors for S and P
S = tf.constant([1.0, 2.0])
P = tf.constant([0.0, 3.14159/2])  # 0 and pi/2 radians

# Original method
SP = tf.cast(S, tf.complex64) * tf.math.exp(1j * tf.cast(P, tf.complex64))

# Alternative method
real_part = S * tf.cos(P)
imag_part = S * tf.sin(P)
complex_tensor = tf.complex(real_part, imag_part)

test = tf.complex(S, P)

# Check if they are the same
equal = tf.reduce_all(tf.math.equal(SP, complex_tensor))

# Evaluate the tensors
print("SP:", SP.numpy())
print("Complex tensor:", complex_tensor.numpy())
print("Are they the same?", equal.numpy())
