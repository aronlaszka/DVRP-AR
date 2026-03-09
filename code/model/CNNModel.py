from model.base.MasterModel import MasterModel


class CNNModel(MasterModel):
    def build_model(self):
        """
        :return: Build a simple CNN model
        """
        import keras
        from model.layers.CNNLayer import CNNLayer
        keras.utils.set_random_seed(self._random_seed)
        custom_inputs = keras.Input(shape=self._input_dim)
        custom_layer = CNNLayer(layer_conf=self._layer_conf["layers"])
        custom_outputs = custom_layer(custom_inputs)
        return keras.Model(custom_inputs, custom_outputs)

    def _generate_fv(self, raw_input):
        """
        :param raw_input: raw input (either state representation or temporary action representation, or both)

        :return: the converted feature vectors:
        provides the number of idle vehicles at each time slot as one dimension array

        Accordingly, feature F can represent as follows
        F(t) = F[t] (total number of vehicles that are free at time slot "t"
        """
        import keras
        return keras.ops.expand_dims(
            keras.ops.sum(
                self.tensor(
                    self._get_availabilities(raw_input, allow_fraction=True)
                ), axis=0
            ),
            axis=-1
        )
