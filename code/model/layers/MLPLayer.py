import keras


class MLPLayer(keras.layers.Layer):
    """
        Custom implementation of Multi-Layer Perceptron (MLP) using Keras
    """

    def __init__(self, layer_conf, normalize=False, **kwargs):
        """
        :param layer_conf: provides the configuration of layers as a list,
         with keys 'neurons', 'activation' and 'drop_out_rate'
        :param normalize: when enable this will normalize the inputs
        """
        super(MLPLayer, self).__init__(**kwargs)
        self.layer_conf = layer_conf
        self.normalize = normalize
        self.dense_layers = []
        self.dropout_layers = {}
        self.normalization_layer = keras.layers.LayerNormalization()
        for i, layer in enumerate(self.layer_conf[:-1]):
            self.dense_layers.append(keras.layers.Dense(layer["neurons"], activation=layer["activation"]))
            if "drop_out_rate" in layer.keys():
                if layer["drop_out_rate"] > 0:
                    self.dropout_layers[i] = keras.layers.Dropout(rate=layer["drop_out_rate"])
        # this defines the output layer
        last_layer = self.layer_conf[-1]
        self.out_layer = keras.layers.Dense(last_layer["neurons"], activation=last_layer["activation"])

    def call(self, inputs, training=False):
        processed_outputs = inputs
        if self.normalize:
            processed_outputs = self.normalization_layer(processed_outputs)
        for i, dense_layer in enumerate(self.dense_layers):
            processed_outputs = dense_layer(processed_outputs)
            if i in self.dropout_layers.keys():
                processed_outputs = self.dropout_layers[i](processed_outputs, training=training)
        return self.out_layer(processed_outputs)

    def __call__(self, inputs, training=False):
        return self.call(inputs, training=training)

    def get_config(self):
        return {
            "layer_conf": self.layer_conf,
            "normalize": self.normalize
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
