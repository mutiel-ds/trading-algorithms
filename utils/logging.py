import os
from logging import getLogger, Logger, Formatter, LogRecord, StreamHandler

class LogFormatter(Formatter):
    def format(self, record: LogRecord) -> str:
        project_root: str = os.path.abspath(path=os.getcwd())
        relative_path: str = os.path.relpath(record.pathname, project_root)
        record.relpath = relative_path.replace("\\", "/")
        return super().format(record=record)

def get_logger(name: str = __name__) -> Logger:
    logger: Logger = getLogger(name=name)
    logger.setLevel(level="DEBUG")

    if not logger.handlers:
        handler: StreamHandler = StreamHandler()
        formatter: LogFormatter = LogFormatter(
            fmt="[%(asctime)s | %(levelname)s] %(relpath)s : %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(fmt=formatter)
        logger.addHandler(hdlr=handler)

    return logger