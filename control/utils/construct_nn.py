import tensorflow as tf


def construct_nn(env, config, initialize=False):
    target_net = tf.keras.Sequential()
    target_net.add(tf.keras.layers.InputLayer(input_shape=env.observation_space.shape))
    for i, layer_size in enumerate(config.layers):
        target_net.add(tf.keras.layers.Dense(layer_size, activation='relu'))
        if config.batch_normalization:
            target_net.add(tf.keras.layers.BatchNormalization())
    target_net.add(tf.keras.layers.Dense(env.action_space.n, activation='linear'))
    target_net.build()
    target_net.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

    return target_net
