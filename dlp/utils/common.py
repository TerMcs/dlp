import importlib
import logging


def init_obj_cls(string_def):
    string_parts = string_def.split(".")
    obj_cls = getattr(importlib.import_module(".".join(string_parts[:-1])), string_parts[-1])
    return obj_cls


def init_obj(string_def, params):
    obj_cls = init_obj_cls(string_def)
    if params is None:
        params = {}
    return obj_cls(**params)

def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:

    logger = logging.getLogger(name)
    logger.setLevel(level)

    return logger

