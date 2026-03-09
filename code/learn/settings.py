def set_threads(number_of_threads=1, log_changes=False):
    """
    :param number_of_threads: number of number_of_threads
    :param log_changes: log changes to terminal
    :return: configure number of number_of_threads for to avoid unlimited number_of_threads initialization
    """
    import os
    import multiprocessing

    number_of_threads = min(max(1, number_of_threads), multiprocessing.cpu_count())

    thread_env_variables = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]
    for thread_env_var in thread_env_variables:
        os.environ[thread_env_var] = str(number_of_threads)
        if log_changes:
            from common.logger import logger
            logger.info(f"Environment variable {thread_env_var} is set to: " + str(number_of_threads))


def get_gpu_memory(gpu_index=-1):
    """
    :param gpu_index: index of the GPU device
    :return: return the memory of NVIDIA GPUs
    """
    import subprocess as sp

    try:
        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_free_values = [int(x.split()[0]) for _, x in enumerate(memory_free_info)]
        if gpu_index >= 0:
            return memory_free_values[gpu_index]
        return memory_free_values
    except sp.CalledProcessError:
        return []


def set_physical_device(
        number_of_threads=8,
        use_gpu=False,
        minimum_required_gpu_memory=8192,
        sleep_duration=15
):
    """
    :param number_of_threads: determine the maximum number of inter and intra threads used by the DNN backends
    :param use_gpu: indicates whether to use GPU or not
    :param minimum_required_gpu_memory: provides the minimum required GPU memory if gpu is chosen
    :param sleep_duration: sleep duration in seconds

    # this function first identify the Deep-Neural Network (DNN) backend is available and set the KERAS_BACKEND to
    one of the first available backend. If no DNN backend is available, then the program terminates with error code -1.

    # once the backend is determined then the inter and intra threads are set to the number specified by 'threads'

    # finally, the function configures the physical device and memory (if applicable)
    """
    import os
    from time import sleep
    from importlib.util import find_spec
    from common.logger import logger
    from common.types import DNNBackend

    # automatically figure out keras backend and updated KERAS_BACKEND environment variable
    # in the event of multiple backend the first choice is always selected
    dnn_backend = None
    for dnn_backend_choice in [item.value for item in DNNBackend]:
        if find_spec(dnn_backend_choice):
            dnn_backend = dnn_backend_choice
            logger.info(f"Using Deep-Neural-Network Backend {dnn_backend.title()}")
            os.environ["KERAS_BACKEND"] = dnn_backend
            break

    if not dnn_backend:
        raise ModuleNotFoundError(
            "Code requires either 'Tensorflow' or 'Torch' backend, But neither of them installed"
        )

    match dnn_backend:
        case DNNBackend.Tensorflow.value:
            import tensorflow as tf
            tf.config.threading.set_intra_op_parallelism_threads(number_of_threads)
            tf.config.threading.set_inter_op_parallelism_threads(number_of_threads)

            cpus = tf.config.list_physical_devices('CPU')
            gpus = tf.config.list_physical_devices('GPU')

            if not use_gpu:
                logger.info(f"[{dnn_backend.title()}] Using {len(cpus)} CPUs")
                tf.config.set_visible_devices(cpus, 'CPU')
                tf.config.set_visible_devices([], 'GPU')
                return

            if len(gpus) > 0:
                import platform
                if platform.system() == "Darwin":
                    # only support M Series Mac
                    tf.config.set_visible_devices(gpus, 'GPU')
                    logger.info(f"[{dnn_backend.title()}] Using {len(gpus)} GPUs (MacOS-ARM)")
                else:
                    memory_values = get_gpu_memory()
                    set_gpu = False
                    for i, memory_value in enumerate(memory_values):
                        if memory_value > minimum_required_gpu_memory:
                            tf.config.set_visible_devices(gpus[i], 'GPU')
                            tf.config.experimental.set_memory_growth(gpus[i], True)
                            set_gpu = True
                            logger.info(
                                f"[{dnn_backend.title()}] Using GPU-{i} with {memory_value / 1024} GB memory"
                            )
                            sleep(sleep_duration)
                            break

                    if not set_gpu:
                        logger.warning(f"[{dnn_backend.title()}] All GPUs are busy, using {len(cpus)} CPUs")
                        tf.config.set_visible_devices(cpus, 'CPU')
                        tf.config.set_visible_devices([], 'GPU')
            else:
                logger.warning(f"[{dnn_backend.title()}] No GPU is available, using {len(cpus)} CPUs")

        case DNNBackend.Torch.value:
            import torch
            # number of threads for intra process (within same function)
            torch.set_num_threads(number_of_threads)
            # number of threads for inter process (between different functions)
            torch.set_num_interop_threads(number_of_threads)

            if not use_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                torch.cuda.is_available = lambda: False  # redundant but extra protection
                return

            if use_gpu:
                if torch.cuda.is_available():
                    set_gpu = False
                    memory_values = get_gpu_memory()
                    for i, memory_value in enumerate(memory_values):
                        if memory_value > minimum_required_gpu_memory:
                            torch.cuda.set_device(f"cuda:{i}")
                            set_gpu = True
                            sleep(sleep_duration)
                            logger.info(
                                f"[{dnn_backend.title()}] Using GPU-{i} with {memory_value / 1024} GB memory"
                            )
                            break

                    if not set_gpu:
                        logger.warning(f"[{dnn_backend.title()}] All GPUs are busy, using CPUs")
                elif torch.backends.mps.is_available():
                    logger.info(f"[{dnn_backend.title()}] Using GPUs (MacOS-ARM)")
                else:
                    logger.warning(f"[{dnn_backend.title()}] No GPU is available, using CPUs")
