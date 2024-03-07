from pathlib import Path
import yaml
from racoons import logger


def configurable(func):
    """Wraps keyword arguments from configuration."""
    def wrapper(*args, **kwargs):
        """Injects configuration keywords."""
        config = kwargs.get("config")
        if config is None:
            logger.info("No config file loaded. Use default parameters.")
            return func(*args, **kwargs)
        elif isinstance(config, dict):
            logger.info("Using configs from dict")
        else:
            with open(config, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration file {config}")
        fname = Path(func.__globals__['__file__']).name
        conf = config[fname][func.__qualname__]
        conf.update(kwargs)
        return func(*args, **conf)
    return wrapper