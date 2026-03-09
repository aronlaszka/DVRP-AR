import keras

from model.base.MasterModel import MasterModel


class KAN(keras.models.Model):
    """
        Custom implementation of KAN Model using Keras
        Original Code: https://github.com/KindXiaoming/pykan (in Pytorch)
    """

    def __init__(
            self,
            layer_neurons,
            activation,
            grid=5,
            k=3,
            noise_scale=0.1,
            noise_scale_base=0.1,
            grid_eps=0.02,
            grid_range=None,
            sp_trainable=True,
            sb_trainable=True,
            bias_trainable=True,
            **kwargs
    ):
        """
        :param layer_neurons: provides the configuration of layers as list, with keys 'neurons'
        :param activation: provides the activation function
        """
        if grid_range is None:
            grid_range = [-1, 1]

        self.layer_neurons = layer_neurons
        self.depth = len(layer_neurons) - 1
        self.activation = activation
        self.grid = grid
        self.k = k

        self.biases = []
        self.act_fun = []

        for l_idx in range(self.depth):
            # splines
            scale_base = 1 / keras.ops.sqrt(
                layer_neurons[l_idx]
            ) + (
                                 keras.random.normal(
                                     shape=(layer_neurons[l_idx] * layer_neurons[l_idx + 1],)
                                 ) * 2 - 1
                         ) * noise_scale_base
            from model.layers.KANLayer import KANLayer
            sp_batch = KANLayer(
                input_dim=layer_neurons[l_idx],
                output_dim=layer_neurons[l_idx + 1],
                activation=activation,
                num=grid,
                k=k,
                noise_scale=noise_scale,
                scale_base=scale_base,
                scale_sp=1.,
                grid_eps=grid_eps,
                grid_range=grid_range,
                sp_trainable=sp_trainable,
                sb_trainable=sb_trainable
            )
            self.act_fun.append(sp_batch)

            # bias
            self.biases.append(
                keras.Variable(initializer="zeros", shape=(layer_neurons[l_idx + 1],), trainable=bias_trainable)
            )
            super(KAN, self).__init__(**kwargs)

    def call(self, inputs, training=False):
        x = inputs
        if inputs.shape[0]:
            for l_idx in range(self.depth):
                x = self.act_fun[l_idx](x) + self.biases[l_idx]
        return x

    def __call__(self, inputs, training=False):
        return self.call(inputs, training=training)

    def get_config(self):
        return {
            "layer_neurons": self.layer_neurons,
            "activation": self.activation,
            "grid": self.grid,
            "k": self.k
        }

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls(**config, **kwargs)


class KANModel(MasterModel):
    def build_model(self):
        """
        :return: Build a simple KAN model
        """
        import keras
        keras.utils.set_random_seed(self._random_seed)
        hidden_widths = self._layer_conf.get("widths", [2 * self._input_dim + 1])
        widths = [self._input_dim] + list(hidden_widths) + [1]

        return KAN(
            layer_neurons=widths,
            activation=keras.activations.silu,
            grid=self._layer_conf.get("grid", 5),
            k=self._layer_conf.get("k", 3),
            noise_scale=self._layer_conf.get("noise_scale", 0.1),
            grid_eps=self._layer_conf.get("grid_eps", 0.02),
            grid_range=self._layer_conf.get("grid_range", [-1, 1]),
            sp_trainable=self._layer_conf.get("sp_trainable", True),
            sb_trainable=self._layer_conf.get("sb_trainable", True),
            bias_trainable=self._layer_conf.get("bias_trainable", True)
        )

    def load_model(self, model_dir, episode, **kwargs):
        """
        :param model_dir: directory containing model files
        :param episode: episode number
        :param kwargs: optional key-word arguments such as
                       - alternate_backend (incase model is trained in different dnn backend)
                       - exit_on_failure (where the program terminates if it is unable to load the model)

        load the weights in h5 file
        load any custom parameters or variables that stored in the pickle format
        load any normalization configuration if found in the models directory
        """
        import keras
        from common.general import file_exists
        from common.logger import logger
        current_backend = keras.config.backend()
        alternate_backend = kwargs['alternate_backend'] if 'alternate_backend' in kwargs else current_backend

        formatted_file_names = {
            key: self._file_names[key].format(model_dir=model_dir, episode=episode).replace(
                current_backend, alternate_backend
            )
            for key in self._file_names.keys()
        }

        load_success = False
        availability_cond = all(
            [
                file_exists(file_name) for key, file_name in formatted_file_names.items()
                if key in ["weight"]
            ]
        )
        if availability_cond:
            try:
                self._model = self.build_model()
                self._model.load_weights(formatted_file_names["weight"])
                self._load_norms(model_dir, episode)
                if current_backend == alternate_backend:
                    logger.info(f"{self.__class__.__name__} loaded successfully from {model_dir} (episode {episode})")
                else:
                    logger.warning(
                        f"{self.__class__.__name__} loaded successfully from {model_dir} (episode {episode}),"
                        f"but model is trained in {alternate_backend} backend and "
                        f"loaded with {current_backend} backend"
                    )
                load_success = True
            except Exception as e:
                self._model = self.build_model()
                logger.error(
                    f"Loading {self.__class__.__name__} from {formatted_file_names['weight']} failed !, exception {e}"
                )
        else:
            if 'exit_on_failure' in kwargs:
                if kwargs['exit_on_failure']:
                    raise FileNotFoundError(f"Weight file {formatted_file_names['weight']} is missing !!!")
            if not file_exists(formatted_file_names["weight"]):
                logger.error(f"Weight file {formatted_file_names['weight']} is missing !!!")
        return load_success
