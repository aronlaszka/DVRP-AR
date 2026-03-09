class DNNBackendNotSupported(Exception):
    def __init__(self, dnn_backend="", func_name="", message=None):
        self.message = message
        if self.message is None:
            self.message = (f"Support for using function {func_name.__name__} not available "
                            f"for Deep-Learning Backend {dnn_backend}")

    def __str__(self):
        return self.message
