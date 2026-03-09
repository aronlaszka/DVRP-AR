import keras


class KANLayer(keras.layers.Layer):
    """
        Custom implementation of KAN Layer Model using Keras
        Original Code: https://github.com/KindXiaoming/pykan (in Pytorch)
    """

    def __init__(
            self,
            input_dim,
            output_dim,
            activation,
            num=5,
            k=3,
            noise_scale=0.1,
            scale_base=1.0,
            scale_sp=1.0,
            grid_eps=0.02,
            grid_range=None,
            sp_trainable=True,
            sb_trainable=True,
            **kwargs
    ):
        """
        :param input_dim: input dimension
        :param output_dim: output dimension
        :param activation: activation function
        """
        super(KANLayer, self).__init__(**kwargs)
        from learn.util_func import curve2coef
        if grid_range is None:
            grid_range = [-1, 1]

        self.size = size = output_dim * input_dim
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.num = num
        self.k = k

        self.grid = keras.ops.einsum(
            'i,j->ij', keras.ops.ones(size),
            keras.ops.linspace(grid_range[0], grid_range[1], num=num + 1)
        )
        self.grid = keras.Variable(self.grid, trainable=False)
        noises = (keras.random.uniform(shape=(size, self.grid.shape[1])) - 1 / 2) * noise_scale / num
        self.coef = keras.Variable(curve2coef(self.grid, noises, self.grid, k))
        if isinstance(scale_base, float):
            self.scale_base = keras.Variable(keras.ops.ones(size) * scale_base, trainable=sb_trainable)
        else:
            self.scale_base = keras.Variable(scale_base, trainable=sb_trainable)
        self.scale_sp = keras.Variable(keras.ops.ones(size) * scale_sp, trainable=sp_trainable)
        self.activation = activation
        self.mask = keras.Variable(keras.ops.ones(size), trainable=False)
        self.grid_eps = grid_eps
        self.lock_counter = 0
        self.lock_id = keras.ops.zeros(size)

    def call(self, inputs, training=False):
        from learn.util_func import coef2curve
        batch = inputs.shape[0]
        x = keras.ops.transpose(
            keras.ops.reshape(
                keras.ops.einsum(
                    'ij,k->ikj', inputs, keras.ops.ones(self.output_dim)
                ),
                (batch, self.size)
            ), [1, 0]
        )
        base = keras.ops.transpose(self.activation(x), axes=[1, 0])  # shape (batch, size)
        y = coef2curve(
            x_eval=x,
            grid=self.grid[:self.size],
            coef=self.coef[:self.size],
            k=self.k
        )  # shape (size, batch)
        y = keras.ops.transpose(y, [1, 0])  # shape (batch, size)
        y = keras.ops.expand_dims(self.scale_base, axis=0) * base + keras.ops.expand_dims(self.scale_sp, axis=0) * y
        y = self.mask[None, :] * y
        return keras.ops.sum(
            keras.ops.reshape(y, (batch, self.output_dim, self.input_dim)), axis=2
        )

    def __call__(self, inputs, training=False):
        return self.call(inputs)

    def get_config(self):
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "activation": self.activation,
            "num": self.num,
            "k": self.k
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
