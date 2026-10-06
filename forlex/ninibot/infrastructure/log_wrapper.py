import logging
import os
from collections.abc import Callable, Iterable

LOG_FORMAT = "%(asctime)s %(message)s"
DEFAULT_LEVEL = logging.DEBUG
MAIN_LOG = "main"
ERROR_LOG = "error"

type LogMessage = Callable[[str, str], None]


class LogWrapper:

    PATH = './logs'

    def __init__(self, name, mode="w", directory=PATH):
        os.makedirs(directory, exist_ok=True)
        self.filename = f"{directory}/{name}.log"
        self.logger = logging.getLogger(name)
        self.logger.setLevel(DEFAULT_LEVEL)

        file_handler = logging.FileHandler(self.filename, mode=mode)
        formatter = logging.Formatter(LOG_FORMAT, datefmt='%Y-%m-%d %H:%M:%S')

        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.logger.info(f"LogWrapper init() {self.filename}")


class BotLogs:
    def __init__(self, pairs: Iterable[str], directory: str = LogWrapper.PATH) -> None:
        names = [*pairs, ERROR_LOG, MAIN_LOG]
        self._loggers = {
            name: LogWrapper(name, directory=directory).logger for name in names
        }

    def message(self, msg: str, key: str) -> None:
        self._loggers[key].debug(msg)
