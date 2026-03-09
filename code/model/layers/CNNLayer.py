import keras


class CNNLayer(keras.layers.Layer):
    """
       Simple Convolutional Neural Network-based Layer
    """

    def __init__(self, layer_conf, dimension=1, **kwargs):
        super(CNNLayer, self).__init__(**kwargs)
        from model.layers.MLPLayer import MLPLayer
        self.layer_conf = layer_conf
        self.dimension = dimension
        self.conv_class_map = {
            1: keras.layers.Conv1D,
            2: keras.layers.Conv2D,
            3: keras.layers.Conv3D
        }
        self.conv = {}
        i = 1
        while f"conv{i}" in self.layer_conf:
            self.conv[i] = self.conv_class_map[self.dimension](**layer_conf[f"conv{i}"])
            i += 1

        if "final_mlp" in self.layer_conf:
            self.flatten = keras.layers.Flatten()
            self.mlp = MLPLayer(layer_conf=layer_conf["final_mlp"], normalize=False)

    def call(self, inputs, training=False):
        conv_out = inputs
        for i, conv_layer in self.conv.items():
            conv_out = conv_layer(conv_out)
        if hasattr(self, "mlp"):
            flatten_output = self.flatten(conv_out)
            return self.mlp(flatten_output, training=training)
        return keras.ops.sum(conv_out, axis=1)

    def __call__(self, inputs, training=False):
        return self.call(inputs, training=training)

    def get_config(self):
        return {
            "layer_conf": self.layer_conf,
            "dimension": self.dimension
        }

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls(**config)
