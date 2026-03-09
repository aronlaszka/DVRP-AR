import logging
from datetime import datetime


class CommandLineFormatter(logging.Formatter):
    red = "\x1b[31;21m"
    green = "\x1b[32;21m"
    yellow = "\x1b[33;21m"
    blue = "\x1b[34;21m"
    magenta = "\x1b[35;21m"
    cyan = "\x1b[36;21m"
    bright_red = "\x1b[91;21m"
    orange = "\x1b[93;21m"
    reset = "\x1b[0m"
    format = "%(message)s"
    date_fmt = '%Y-%m-%d %H:%M:%S'
    time_fmt = cyan + "[%(asctime)s] "
    level_fmt = "[%(levelname)s] "

    FORMATS = {
        logging.INFO: time_fmt + blue + level_fmt + reset + format,
        logging.DEBUG: time_fmt + yellow + level_fmt + reset + format,
        logging.WARNING: time_fmt + orange + level_fmt + reset + format,
        logging.ERROR: time_fmt + red + level_fmt + reset + format,
        logging.CRITICAL: time_fmt + bright_red + level_fmt + reset + format,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt=self.date_fmt)
        return formatter.format(record)


class LogFormatter(logging.Formatter):
    fmt = "[%(asctime)s] [%(levelname)s] %(message)s"
    date_fmt = '%Y-%m-%d %H:%M:%S'

    def format(self, record):
        formatter = logging.Formatter(self.fmt, datefmt=self.date_fmt)
        return formatter.format(record)


class Logger(object):
    def __init__(
            self,
            log_directory="logs",
            log_prefix="sch",
            log_suffix="abc",
            logging_level=logging.INFO,
            log_buffer_size=1024,
            log_flush_logging_level=logging.ERROR,
            log_flush_on_close=True,
            log_to_command_line=False
    ):
        """
        :param log_directory: directory to which the logs are stored
        :param log_prefix: prefix to filter out the logs within the log-directory
        :param log_suffix: suffix to filter out the logs within the log-directory
        :param logging_level: logging level of Logger instance
        :param log_buffer_size: size of buffer to be used before flushing
        :param log_flush_logging_level: logging level before flushing the buffered logs
        :param log_flush_on_close: flush the buffer when the logger is closed
        :param log_to_command_line: if enabled logging to the command-line as well
        """
        self._instance = logging.getLogger(f"{log_prefix.upper()}-{log_suffix.upper()}")
        self._loc_directory = log_directory
        self._instance_prefix = log_prefix
        self._instance_suffix = log_suffix
        self._log_buffer_size = log_buffer_size
        self._log_flush_on_close = log_flush_on_close
        self._log_flush_logging_level = log_flush_logging_level
        self._instance_curr_time = int(datetime.now().timestamp())
        self._instance_log_file_name = (
            f"{self._loc_directory}/"
            f"{self._instance_prefix.lower().replace('-', '/')}/"
            f"{self._instance_suffix.lower().replace('-', '/')}/{self._instance_curr_time}.log"
        )
        self._instance.setLevel(level=logging_level)
        self._file_handler = None
        self._command_line_handler = None
        self._add_file_handler(logging_level=logging_level)
        if log_to_command_line:
            self._add_cmd_handler(logging_level=logging_level)

    def _add_cmd_handler(self, logging_level=logging.INFO):
        """
        :param logging_level: logging level for the message
        :return: create a command-line to log the message based on the logging level
        """
        from logging.handlers import MemoryHandler
        self._command_line_handler = logging.StreamHandler()
        self._command_line_handler.setLevel(logging_level)
        self._command_line_handler.setFormatter(CommandLineFormatter())
        self.add_handler(self._command_line_handler)

    def _add_file_handler(self, logging_level=logging.INFO):
        """
        :param logging_level: logging level for the message
        :return: create a persistent-file handler to log the message based on the logging level
        """
        from logging.handlers import MemoryHandler
        from common.general import create_dir
        create_dir(self._instance_log_file_name, allow_file_path=True)
        self._file_handler = logging.FileHandler(self._instance_log_file_name)
        memory_handler = MemoryHandler(
            self._log_buffer_size,
            flushLevel=self._log_flush_logging_level,
            target=self._file_handler,
            flushOnClose=self._log_flush_on_close
        )
        self._file_handler.setLevel(logging_level)
        self._file_handler.setFormatter(LogFormatter())
        self.add_handler(self._file_handler)
        self.add_handler(memory_handler)

    def get_file_handler(self):
        """
        :return: file handler object
        """
        return self._file_handler

    def add_handler(self, handler):
        """
        :param handler: logging handler which is a subclass from the parent class `Handler'
        """
        from logging import Handler
        if isinstance(handler, Handler):
            self._instance.addHandler(handler)
        else:
            self._instance.error(
                f"Unable to add the handler of class {handler.__class__.__name__}"
            )

    def get_instance(self):
        """
        :return: logger instance
        """
        return self._instance

    def get_log_file_name(self):
        """
        :return: the name of the log file for easy tracking of logs
        """
        return self._instance_log_file_name

    def get_prefix(self):
        """
        :return: the logger prefix
        """
        return self._instance_prefix

    def get_suffix(self):
        """
        :return: the logger suffix
        """
        return self._instance_suffix


logger = Logger(
    log_directory="logs",
    log_prefix="client",
    log_suffix="ovwab",
    logging_level=logging.INFO,
    log_buffer_size=1024,
    log_flush_logging_level=logging.ERROR,
    log_flush_on_close=True,
    log_to_command_line=True,
).get_instance()

