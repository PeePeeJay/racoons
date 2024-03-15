from pathlib import Path
import yaml


def configurable(func):
    """Wraps keyword arguments from configuration."""
    def wrapper(*args, **kwargs):
        """Injects configuration keywords."""
        config = kwargs.get("config")
        if config is None:
            return func(*args, **kwargs)
        elif isinstance(config, dict):
            print(f"Use config from dict")
        else:
            with open(config, 'r') as f:
                config = yaml.safe_load(f)
        fname = Path(func.__globals__['__file__']).name
        conf = config[fname][func.__qualname__]
        conf.update(kwargs)
        return func(*args, **conf)
    return wrapper