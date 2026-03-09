class KerasBaseModel:
    def __init__(self, architecture, file_names, configs):
        self._dtype = "float32"
        self._device = "cuda" if configs.get("use_gpu", False) else "cpu"
        self._random_seed = configs.get("random_seed", 0)
        self._threads = configs.get("threads", 1)
        self._input_dim = architecture["dimension"]["input"]
        self._feature_code = architecture["dimension"]["feature_code"]
        self._polynomial_dim = architecture["dimension"]["polynomial"]
        self._file_names = file_names
        self._layer_conf = architecture["layer_configuration"]
        self._norm_config = architecture.get("normalization_config", {})
        self._model_controls = configs.get("model_controls", {})
        self._model = self.build_model()
        self._plot_model = configs.get("plot_model", False)

    def build_model(self):
        raise NotImplementedError

    def get_model(self):
        return self._model

    def tensor(self, raw_input):
        """
        :param raw_input: raw input either single value, or arrays
        :returns: tensor object match with deep-learning backend where the value is equal to the raw_input
        """
        if hasattr(self, '_scikit_mode'):
            # this will avoid unwanted conversion of numpy array to tensor array
            if self._scikit_mode:
                import numpy as np
                return np.asfortranarray(raw_input)

        import keras
        return keras.ops.convert_to_tensor(raw_input, dtype=self._dtype)

    def inference(self, inputs, training=False, squeeze=False, add_noise=False, epsilon=1.0, std_dev=0.1):
        """
        :param inputs: input features
        :param training: indicates whether in training mode or not
        :param squeeze: squeeze unwanted last dimension (temporary fix)
        :param add_noise: add noise to the prediction or not
        :param epsilon: epsilon value for the noise (applicable only if the noise is added to the prediction)
        :param std_dev: standard deviation of the noise (applicable only if the noise is added to the prediction)
        return inferences from the model based on the given input (in inputs) as a tensor object
        """
        import keras
        result = self._model(inputs, training=training)
        if add_noise:
            result += keras.random.normal(result.shape, mean=0, stddev=std_dev * epsilon)
        if squeeze:
            if len(result.shape) > 1:
                result = keras.ops.squeeze(result)
        return result

    def get_trainable_weights(self):
        """
        :return: all parameters as list (used for computing gradient)
        """
        return self._model.trainable_weights

    def get_norm(self, feature_name):
        """
        :param feature_name: Name of the input feature
        :return: return the factor by which the input feature is multiplied
        """
        from common.general import time_unit_conversion
        if feature_name.endswith("_time"):
            if "input_unit" in self._norm_config and "output_unit" in self._norm_config:
                if feature_name in self._norm_config["input_unit"] and \
                        feature_name in self._norm_config["output_unit"]:
                    input_unit = self._norm_config["input_unit"][feature_name]
                    output_unit = self._norm_config["output_unit"][feature_name]
                    return time_unit_conversion(input_unit, output_unit)

        if "default" in self._norm_config and feature_name in self._norm_config["default"]:
            return self._norm_config["default"][feature_name]
        raise ValueError(f"Normalization does not exists for {feature_name}")

    def load_model(self, model_dir, episode, **kwargs):
        """
        :param model_dir: directory containing model files
        :param episode: episode number
        :param kwargs: optional key-word arguments such as
                       - alternate_backend (incase model is trained in different dnn backend)
                       - exit_on_failure (where the program terminates if it is unable to load the model)

        load the model that the keras format
        load the weights that stored in h5 format and
        load any custom parameters or variables that stored in the pickle format
        load any normalization configuration if found in the models directory
        """
        import keras
        from common.general import file_exists
        from common.logger import logger

        current_backend = keras.config.backend()
        alternate_backend = kwargs['alternate_backend'] if 'alternate_backend' in kwargs else current_backend

        formatted_file_names = {
            key: self._file_names[key].format(
                model_dir=model_dir, episode=episode
            ).replace(current_backend, alternate_backend)
            for key in self._file_names.keys()
        }
        availability_cond = all(
            [
                file_exists(file_name) for key, file_name in formatted_file_names.items()
                if key in ["model", "weight"]
            ]
        )
        load_success = False
        if availability_cond:
            try:
                if hasattr(self._model, 'load_model_alternate'):
                    self._model.load_model_alternate(formatted_file_names["model"])
                else:
                    from learn.util_func import (mean_error, mean_bias_error,
                                                 mean_prediction, variance_prediction, mean_true, variance_true)
                    self._model = keras.models.load_model(
                        formatted_file_names["model"], custom_objects={
                            "mean_error": mean_error,
                            "mean_bias_error": mean_bias_error,
                            "mean_prediction": mean_prediction,
                            "variance_prediction": variance_prediction,
                            "mean_true": mean_true,
                            "variance_true": variance_true
                        }
                    )
                self._model.load_weights(formatted_file_names["weight"])
                if current_backend == alternate_backend:
                    logger.info(f"{self.__class__.__name__} loaded successfully from {model_dir} (episode {episode})")
                else:
                    logger.warning(
                        f"{self.__class__.__name__} loaded successfully from {model_dir} (episode {episode}), "
                        f"but model is trained in {alternate_backend} backend and "
                        f"loaded with {current_backend} backend"
                    )
                self._load_norms(model_dir, episode)
                load_success = True
            except Exception as e:
                self._model = self.build_model()
                logger.error(
                    f"Loading {self.__class__.__name__} from {model_dir} (episode {episode}) failed !, exception {e}"
                )
        else:
            if 'exit_on_failure' in kwargs:
                if kwargs['exit_on_failure']:
                    err_str = ""
                    if not file_exists(formatted_file_names["model"]):
                        err_str += f"Model file {formatted_file_names['model']} is missing !!!"

                    if not file_exists(formatted_file_names["weight"]):
                        err_str += f"Weights file {formatted_file_names['weight']} is missing !!!"

                    raise FileNotFoundError(err_str)

            if not file_exists(formatted_file_names["model"]):
                logger.error(f"Model file {formatted_file_names['model']} is missing !!!")

            if not file_exists(formatted_file_names["weight"]):
                logger.error(f"Weight file {formatted_file_names['weight']} is missing !!!")
        return load_success

    def _load_norms(self, model_dir, episode):
        """
        :param model_dir: directory containing model files
        :param episode: episode number

        load norms
        """
        import json
        from common.general import file_exists
        from common.logger import logger
        if "norm" in self._file_names and hasattr(self, "_norm_config"):
            norm_config_file_path = self._file_names["norm"].format(model_dir=model_dir, episode=episode)
            if file_exists(norm_config_file_path):
                with open(norm_config_file_path, "r") as norm_config_file:
                    self._norm_config = json.load(norm_config_file)
                    logger.info(f"Loaded normalization configuration from {norm_config_file_path}")

    def save_model(self, output_dir, episode):
        """
        :param output_dir: the directory to store the models
        :param episode: either the current episode number or string representing the current training check point

        save the model in the keras format
        save the weights in h5 format
        save any custom parameters or variables in the pickle format if exists and
        save any normalization .configs in the json format if exists
        """
        import keras
        from common.general import create_dir, file_exists
        from common.logger import logger
        if self._model:
            formatted_file_names = {
                key: self._file_names[key].format(model_dir=output_dir, episode=episode)
                for key in self._file_names.keys()
            }
            create_dir(formatted_file_names["weight"])

            if self._plot_model:
                # we only need to save the image once !!!
                if not file_exists(formatted_file_names["image"]):
                    from importlib.util import find_spec
                    if find_spec("pydot") and find_spec("graphviz"):
                        keras.utils.plot_model(
                            self._model,
                            to_file=formatted_file_names["image"],
                            show_layer_names=True,
                            show_shapes=True,
                            show_layer_activations=True,
                            show_trainable=True,
                            show_dtype=True,
                            expand_nested=True
                        )
                    else:
                        if not find_spec("pydot"):
                            logger.warning("Unable to plot the model, please install pydot")

                        if not find_spec("graphviz"):
                            logger.warning("Unable to plot the model, please install graphviz")

            # we only need to save the model once
            if not file_exists(formatted_file_names["model"]):
                self._model.save(formatted_file_names["model"])
            self._model.save_weights(formatted_file_names["weight"])
            self._save_norms(output_dir, episode)

    def _save_norms(self, output_dir, episode):
        """
        :param output_dir: the directory to store the models
        :param episode: either the current episode number or string representing the current training check point

        save any normalization .configs in the json format if exists
        """
        if "norm" in self._file_names and hasattr(self, "_norm_config"):
            # this is optional only if there is any norm configuration exists
            import json
            if len(self._norm_config) > 0:
                config_file_path = self._file_names["norm"].format(model_dir=output_dir, episode=episode)
                with open(config_file_path, "w") as norm_config_file:
                    json.dump(self._norm_config, norm_config_file, indent=4)

    def get_model_weights(self):
        """
        :return: model weights
        """
        return self._model.get_weights()

    def sync_weights(self, other_estimator, tau=None):
        """
        :param other_estimator: other estimator
        :param tau: soft updates on target weight

        sync the estimators weight with other estimator
        """
        model_theta = other_estimator.get_model_weights()
        target_model_theta = self.get_model_weights()
        if tau:
            for i, (target_weight, model_weight) in enumerate(zip(target_model_theta, model_theta)):
                target_weight = target_weight * (1 - tau) + model_weight * tau
                target_model_theta[i] = target_weight
            self._model.set_weights(target_model_theta)
        else:
            self._model.set_weights(model_theta)
