import keras
from keras import backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

config = {
    'inputs': 28*28,
    'layer_sizes': [600, 500, 400, 300, 200, 150, 100, 75, 50, 30, 20, 10, 4],
    'batch_size': 64,
    'bottleneck_cells': 20., # TODO 8.,
    'epochs': 100,
    'label_num': 10, # How many labels there are in the data
    'samples': 20000 # How many data points to predict on in the evaluation phase (unless the test set is used)
}

# Custom loss
# ----------------------

def bottleneck_loss(y_true, y_pred):
    # y_true_prot = tf.stop_gradient(y_true)
    
    loss = 1. - 1./( 1. + tf.math.exp(-25.*((y_pred-.5)**2.)) )
    loss *= 2.
    
    # loss = 1. - tf.math.cos(y_pred * math.pi * .8)
    
    # loss = 0.
    
    return tf.reduce_mean(loss)


# Construct the model
# ----------------------

inputs = keras.layers.Input(shape=(config['inputs'],), dtype='float32')
x = inputs

for ix, size in enumerate(config['layer_sizes']):
    x = keras.layers.Dense(size, name="encode/dense{}".format(ix))(x)
    x = keras.layers.Activation('sigmoid')(x)
    x = keras.layers.BatchNormalization(name="encode/bn{}".format(ix))(x)
x = keras.layers.Activation('sigmoid')(x) # to ensure bottleneck signal is 0-1

bottleneck = x
quantised_bottleneck = keras.layers.Lambda((lambda y: tf.math.floor(y*(config['bottleneck_cells']-.0001))/config['bottleneck_cells'] ))(x)

decoder_input = keras.layers.Input(shape=(config['layer_sizes'][-1],), dtype='float32')
dx = decoder_input
for ix, size in enumerate(reversed(config['layer_sizes'])):
    dx = keras.layers.Dense(size, name="decode/dense{}".format(ix))(dx)
    dx = keras.layers.Activation('sigmoid')(dx)
    dx = keras.layers.BatchNormalization(name="decode/bn{}".format(ix))(dx)
dx = keras.layers.Dense(config['inputs'], name="decode/denseFinal")(dx)
decoder_model = keras.models.Model(inputs=[decoder_input], outputs=[dx])

decoded = decoder_model(bottleneck)
quantised_decoded = decoder_model(quantised_bottleneck)

# Name the outputs
decoded = keras.layers.Lambda((lambda y: y), name='decoded')(decoded)
bottleneck = keras.layers.Lambda((lambda y: y), name='bottleneck')(bottleneck)
quantised_decoded = keras.layers.Lambda((lambda y: y), name='quantised_decoded')(quantised_decoded)

model = keras.models.Model(inputs=[inputs], outputs=[decoded, bottleneck, quantised_decoded])
model.compile(
    loss={'decoded': 'mse', 'bottleneck': bottleneck_loss, 'quantised_decoded': 'mse'},
    loss_weights={'decoded': 1., 'bottleneck': .01, 'quantised_decoded': 2.},
    optimizer=keras.optimizers.Adadelta()
)

# Load and prepare the data
# ----------------------

# We don't use the labels; we are interested in how the model classifies the images
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

x_train = np.reshape(x_train, (-1, 28*28))

# Train the model
# ----------------------

# The dummy target for the bottleneck loss is not used
model.fit(x_train, [x_train, np.zeros((x_train.shape[0], 1)), x_train], batch_size=config['batch_size'], epochs=config['epochs'])

# Run a sample of the data through the model
# ----------------------

bottleneck_dimensions = config['layer_sizes'][-1]

if True: # sample from training data
    samples = x_train[0:config['samples']]
    label_index = y_train[0:config['samples']]
else: # use the test set
    samples = y_train
    label_index = y_test

sample_size = samples.shape[0]
code = model.predict(samples)
code = code[1] # select the bottleneck output

# Plot where the sample data points are in the bottleneck "cube", colored by their labels

for i in range(bottleneck_dimensions):
    for j in range(i+1, bottleneck_dimensions):
        plt.scatter(code[:,i], code[:,j], s=.4, c=label_index)
        plt.title("Bottleneck scatter Dimensions i={} j={}".format(i, j))
        plt.savefig("out/bottleneck_scatter_4fq20_{}_{}.png".format(i, j), dpi=200)

# Collect stats on what kind of data points are in the clusters (corners)

cluster_mask = np.zeros((bottleneck_dimensions,)) # to convert bitmask to index
for i in range(bottleneck_dimensions):
    cluster_mask[i] = 2**i # should mirror the above {MASK}

# This holds the cluster (corner) index for each data point in the sample
cluster_index = np.sum(np.floor(code*1.9999) * cluster_mask, axis=1)

cluster_sizes = np.zeros((2**bottleneck_dimensions,))
cross_analysis = np.zeros((config['label_num'], 2**bottleneck_dimensions))
for i in range(sample_size):
    cluster_sizes[int(cluster_index[i])] += 1
    cross_analysis[int(label_index[i]), int(cluster_index[i])] += 1

print("Clusters vs labels")
for j in range(config['label_num']):
    print("L{:6.0f}".format(j), end='')
print("  <- Labels")
for i in range(2**bottleneck_dimensions):
    for j in range(config['label_num']):
        print("{:7.0f}".format(cross_analysis[j,i]), end='')
    print("  Cluster {:2.0f} size {:6.0f}".format(i, cluster_sizes[i]))

# Generate images from the extreme points 
# ----------------------

for cluster in range(2**bottleneck_dimensions):
    input = np.zeros((1, bottleneck_dimensions))
    for i in range(bottleneck_dimensions):
        if (cluster & (2**i)):
            input[0, i] = 1
    decoded = decoder_model.predict(input)[0]
    decoded = np.reshape(decoded, (28, 28))
    plt.imsave("out/decoded_4fq20_{}.png".format(cluster), decoded)
