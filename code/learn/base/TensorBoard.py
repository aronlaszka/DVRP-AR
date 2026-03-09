class TensorBoard:
    """
    To store the statistics related to training process
    """

    def __init__(self, output_dir, random_seed=0):
        import keras
        from common.general import create_dir
        from common.types import DNNBackend

        self.__random_seed = random_seed
        self.__epoch_counters = {}
        self.__writers = {}
        self.__csv_file_names = {}
        self.__output_dir = output_dir
        create_dir(f"{self.__output_dir}/summary/")
        if keras.config.backend() == DNNBackend.Torch.value:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(self.__output_dir)

        self._logging_keys = []
        self.reset_board()

    def reset_board(self):

        self.__epoch_counters = {}
        self.__writers = {}
        self.__csv_file_names = {}

        for key in self._logging_keys:
            self._create_key(key)

    def _create_key(self, key):
        """
        :param key: the name of the parameter/variable/metrics that to be logged
        """
        import os
        import keras
        from common.general import create_dir
        from common.types import DNNBackend
        self.__epoch_counters[key] = 0
        if keras.config.backend() == DNNBackend.Tensorflow.value:
            import tensorflow as tf
            self.__writers[key] = tf.summary.create_file_writer(
                f"{self.__output_dir}/logs/{key}_{self.__random_seed}"
            )

        self.__csv_file_names[key] = f"{self.__output_dir}/summary/{self.__random_seed}/{key}.csv"

        create_dir(self.__csv_file_names[key])
        with open(self.__csv_file_names[key], "w+") as summary_file:
            summary_file.write(f"step,{key}\n")
            summary_file.flush()
            os.fsync(summary_file.fileno())

    def log(self, **kwargs):
        """
        :param kwargs: list of name(or key)-value pairs of the parameter/variable/metrics
        log to the tensorboard, and write to file in human-readable format
        """

        for _key, _value in kwargs.items():
            if _key not in self._logging_keys:
                self._create_key(_key)
                self._logging_keys.append(_key)
            self.__log(_key, _value)

    def __log(self, _key, _value):
        """
        :param _key: the name of the parameter/variable/metrics that to be logged
        :param _value: the value of the parameter/variable/metrics

        log to the tensorboard, and write to file in human-readable format
        """
        import os
        import keras
        from common.types import DNNBackend
        from learn.base.Errors import DNNBackendNotSupported
        backend = keras.config.backend()

        # logging into tensorboard log files
        if backend == DNNBackend.Tensorflow.value:
            import tensorflow as tf
            with self.__writers[_key].as_default():
                tf.summary.scalar(_key.replace("_", "/", 1), _value, step=self.__epoch_counters[_key])
        elif backend == DNNBackend.Torch.value:
            self._writer.add_scalar(_key.replace("_", "/", 1), _value, self.__epoch_counters[_key])
        else:
            raise DNNBackendNotSupported(dnn_backend=backend, func_name="log")

        # writing into human-readable log files
        with open(self.__csv_file_names[_key], "a+") as summary_file:
            summary_file.write(f"{self.__epoch_counters[_key]},{_value}\n")
            summary_file.flush()
            os.fsync(summary_file.fileno())
        self.__epoch_counters[_key] += 1
