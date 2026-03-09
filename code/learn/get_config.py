import copy
import itertools


def get_mlp_configs(_number_of_layers, _number_of_neurons, _drop_out_rates):
    counter = 0
    custom_configs = {}
    base_config = {"layers": ""}
    for drop_out_rate in _drop_out_rates:
        for _number_of_layer in _number_of_layers:
            number_of_neurons_list = []
            for n_l in range(_number_of_layer):
                number_of_neurons_list.append(_number_of_neurons)
            number_of_neuron = [nn_comb for nn_comb in itertools.product(*number_of_neurons_list)]
            for non in number_of_neuron:
                layers = []
                for k in range(_number_of_layer):
                    layer_config = {
                        "neurons": non[k],
                        "activation": "relu",
                        "drop_out_rate": drop_out_rate
                    }
                    layers.append(layer_config)
                layers.append({
                    "neurons": 1,
                    "activation": "linear",
                })
                base_config["layers"] = layers
                custom_configs[counter] = copy.deepcopy(base_config)
                counter += 1
    return custom_configs


mlp_configs = {
    0: get_mlp_configs(
        [3, 2, 1],
        [128, 64, 32],
        [0.2, 0.1, 0.0]
    )
}


def get_cnn_configs(_channels, _kernel_sizes, _number_of_neurons, _drop_out_rates):
    counter = 0
    channel_v = [a for a in itertools.combinations_with_replacement(_channels, 2)]
    custom_configs = {}
    base_config = {
        "layers": {
            "conv1": dict(),
            "conv2": dict(),
            "final_mlp": list(),
        }
    }
    for ch in channel_v:
        for k in _kernel_sizes:
            for neu in _number_of_neurons:
                for drop_out_rate in _drop_out_rates:
                    base_config["layers"]["conv1"] = {
                        "filters": ch[0],
                        "kernel_size": k,
                        "activation": "relu"
                    }
                    base_config["layers"]["conv2"] = {
                        "filters": ch[1],
                        "kernel_size": k,
                        "activation": "relu"
                    }
                    base_config["layers"]["final_mlp"] = [
                        {
                            "activation": "relu",
                            "drop_out_rate": drop_out_rate,
                            "neurons": neu
                        },
                        {
                            "activation": "linear",
                            "neurons": 1
                        }
                    ]
                    custom_configs[counter] = copy.deepcopy(base_config)
                    counter += 1
    return custom_configs


cnn_configs = {
    0: get_cnn_configs(
        [16, 8, 4],
        [2, 3, 4],
        [128, 64],
        [0.0, 0.1]
    )
}


def get_kan_configs(_number_of_layers, _number_of_neurons):
    base_config = {
        "widths": "",
        "grid": 5,
        "k": 3,
        "noise_scale": 0.1,
        "grid_eps": 0.02,
        "grid_range": [
            -1,
            1
        ],
        "sp_trainable": True,
        "sb_trainable": True,
        "bias_trainable": True,
        "custom_parameters": {
            "threshold": 3.0,
            "addition": 3.0
        }
    }

    counter = 0
    custom_configs = {}
    for _number_of_layer in _number_of_layers:
        number_of_neurons_list = []
        for n_l in range(_number_of_layer):
            number_of_neurons_list.append(_number_of_neurons)
        number_of_neuron = [nn_comb for nn_comb in itertools.product(*number_of_neurons_list)]
        for non in number_of_neuron:
            base_config["widths"] = non
            custom_configs[counter] = copy.deepcopy(base_config)
            counter += 1
    return custom_configs


kan_configs = {
    0: get_kan_configs(
        [2, 1],
        [i for i in range(9, 0, -1)],
    )
}

configs = {
    "kan": kan_configs,
    "mlp": mlp_configs,
    "cnn": cnn_configs,
}
