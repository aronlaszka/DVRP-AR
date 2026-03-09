def mean_error(y_true, y_pred):
    import keras
    return keras.ops.mean(y_true - y_pred)


def mean_bias_error(y_true, y_pred):
    import keras
    return keras.ops.mean(y_pred - y_true)


def mean_prediction(_, y_pred):
    import keras
    return keras.ops.mean(y_pred)


def mean_true(y_true, _):
    import keras
    return keras.ops.mean(y_true)


def variance_prediction(_, y_pred):
    import keras
    return keras.ops.var(y_pred)


def variance_true(y_true, _):
    import keras
    return keras.ops.var(y_true)


def get_common_keras_metrics():
    """
    returns some keras metrics and custom keras metrics
    """
    import keras
    return [
        keras.metrics.R2Score(),
        keras.metrics.RootMeanSquaredError(),
        keras.metrics.MeanAbsoluteError(),
        mean_error,
        mean_bias_error,
        mean_prediction,
        variance_prediction,
        mean_true,
        variance_true
    ]


def is_keras_backend_available():
    """
    indicates whether keras is supported by deep neural networks backend
    """
    from common.types import DNNBackend
    from importlib.util import find_spec

    # check whether at-least one of the keras backend available or not
    for dnn_backend_choice in [item.value for item in DNNBackend]:
        if find_spec(dnn_backend_choice):
            return True
    return False


def convert_to_numeral(input_tensor, dtype=float):
    """
    :param input_tensor: input tensor
    :param dtype: output dtype
    :return: return specified by the dtype value (if it is scaler input tensor) otherwise list
    """
    if is_keras_backend_available():
        import keras
        from keras.src.backend.common import KerasVariable
        if keras.ops.is_tensor(input_tensor) or isinstance(input_tensor, KerasVariable):
            if len(input_tensor.shape) == 0:
                return dtype(str(keras.ops.convert_to_numpy(input_tensor)))
            return keras.ops.convert_to_numpy(input_tensor).astype(dtype).tolist()
    return input_tensor


def get_magnitude(input_tensor, order=2):
    """
    :param input_tensor: input tensor
    :param order: norm order (default: 2, 2-norm)
    return get the magnitude of the input tensor (with varying dimension)
    """
    import keras
    flatten_gradients = []
    for sub_tensor in input_tensor:
        if sub_tensor is not None:
            flatten_gradients.extend(keras.ops.reshape(sub_tensor, (-1)))
    return keras.ops.norm(flatten_gradients, ord=order)


def repeat(x_inp):
    """
    :param x_inp: input tensor
    :return: repeated input tensor (based on the number of entries)
    """
    import keras
    x, inp = x_inp
    x = keras.ops.expand_dims(x, 1)
    x = keras.ops.repeat(x, [keras.ops.shape(inp)[1]], axis=1)
    return x


def list_multiply(entries):
    """
    :param entries: list of values

    sequentially multiply each element of the list, if one element is zero return zero
    """
    s = entries[0]
    if s == 0.0:
        return 0.0
    for h in range(1, len(entries)):
        if entries[h] == 0.0:
            return 0.0
        s *= entries[h]
    return s


def polynomial_feature(raw_features, polynomial_dim):
    """
    :param raw_features: list of raw feature vectors
    :param polynomial_dim: polynomial dimension
    """
    import copy
    import itertools
    raw_features_cpy = copy.deepcopy(raw_features)
    for dim in range(2, polynomial_dim + 1):
        svs_x = itertools.combinations_with_replacement(raw_features_cpy, dim)
        for sv_xx in svs_x:
            raw_features.append(list_multiply(list(sv_xx)))
    return raw_features


def train_step_tensorflow(loss_function, loss_function_args, trainable_variables, optimizer):
    """
    :param loss_function: API to compute the loss function
    :param loss_function_args: arguments to pass to the loss function
    :param trainable_variables: variables that needs to be trained
    :param optimizer: optimizer to use
    """
    # import gc
    # import keras
    import tensorflow as tf
    with tf.GradientTape() as tape:
        train_response = loss_function(**loss_function_args)

    gradients = tape.gradient(train_response["loss"], trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    train_response["gradients"] = gradients
    # keras.utils.clear_session()
    # gc.collect()
    return train_response


def train_step_torch(loss_function, loss_function_args, model, trainable_variables, optimizer, use_gpu):
    """
    :param loss_function: API to compute the loss function
    :param loss_function_args: arguments to pass to the loss function
    :param model: model to train
    :param trainable_variables: variables that needs to be trained
    :param optimizer: optimizer to use
    :param use_gpu: whether to use GPU or not
    """
    # import gc
    # import keras
    import torch

    if torch.cuda.is_available():
        with torch.autocast(device_type="cuda" if use_gpu else "cpu"):
            train_response = loss_function(**loss_function_args)
    else:
        train_response = loss_function(**loss_function_args)

    model.get_model().zero_grad()
    trainable_weights = [weight for weight in trainable_variables]

    train_response["loss"].backward()
    gradients = [weight.value.grad for weight in trainable_weights]

    train_response["gradients"] = gradients

    with torch.no_grad():
        optimizer.apply(gradients, trainable_weights)

    # keras.utils.clear_session()
    # gc.collect()
    return train_response


"""
   Implementation of B-Spline using Keras
   Original Code: https://github.com/KindXiaoming/pykan (in Pytorch)
"""


def extend_grid(grid, k_extend=0):
    import keras
    # pad k to left and right
    # grid shape: (batch, grid)
    h = (grid[:, -1:] - grid[:, :1]) / (grid.shape[1] - 1)

    for i in range(k_extend):
        grid = keras.ops.concatenate([grid[:, :1] - h, grid], axis=1)
        grid = keras.ops.concatenate([grid, grid[:, -1:] + h], axis=1)
    return grid


def B_batch(x, grid, k=0, extend=True):
    import keras
    if extend:
        grid = extend_grid(grid, k_extend=k)

    grid = keras.ops.expand_dims(grid, axis=2)
    x = keras.ops.expand_dims(x, axis=1)

    if k == 0:
        value = keras.ops.logical_and(keras.ops.greater_equal(x, grid[:, :-1]), keras.ops.less(x, grid[:, 1:]))
        value = keras.ops.cast(value, 'float32')
    else:
        B_km1 = B_batch(x[:, 0], grid=grid[:, :, 0], k=k - 1, extend=False)
        value = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * B_km1[:, :-1] + (
                grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * B_km1[:, 1:]
    return value


def coef2curve(x_eval, grid, coef, k):
    import keras
    return keras.ops.einsum('ij,ijk->ik', coef, B_batch(x_eval, grid, k))


def curve2coef(x_eval, y_eval, grid, k):
    import keras
    initial_mat = B_batch(x_eval, grid, k)
    mat = keras.ops.transpose(initial_mat, axes=[0, 2, 1])
    y_eval = keras.ops.expand_dims(y_eval, axis=2)
    backend = keras.config.backend()
    if backend == "tensorflow":
        import tensorflow as tf
        return tf.linalg.lstsq(mat, y_eval)[:, :, 0]
    elif backend == "torch":
        import torch
        return torch.linalg.lstsq(mat, y_eval).solution[:, :, 0]
    else:
        from base.Errors import DNNBackendNotSupported
        raise DNNBackendNotSupported(dnn_backend=backend, func_name='curve2coef')
