from model.base.MasterModel import MasterModel


class MLPModel(MasterModel):
    def build_model(self):
        """
        :return: Build a simple MLP model
        """

        import keras
        from model.layers.MLPLayer import MLPLayer
        keras.utils.set_random_seed(self._random_seed)
        inputs = keras.Input(shape=(self._input_dim,))
        mlp_layer = MLPLayer(layer_conf=self._layer_conf["layers"], normalize=False)
        outputs = mlp_layer(inputs)
        return keras.Model(inputs, outputs)
