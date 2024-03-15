import yaml

from racoons.config import configurable


@configurable
def configurable_method(configurable_kwarg=1, **kwargs):
    msg = kwargs.get("msg", "default")
    return configurable_kwarg, msg


class ConfigurableClass:
    def __init__(self, conf):
        self.conf = conf

def test_config(tmp_path):
    # assert configurable_method() == 1

    # config dict
    test_conf = {"test_config.py": {"configurable_method": {"configurable_kwarg": 0, "msg": "test"}}}

    # config yaml
    with open(f"{tmp_path}/test_conf.yaml", "w") as f:
        yaml.dump(test_conf, f)

    # read config as dict
    value, msg = configurable_method(config=test_conf)
    assert value == 0
    assert msg == "test"

    # read config from yaml
    # assert (
    #     configurable_method(
    #         config=f"{tmp_path}/test_conf.yaml"
    #     )
    #     == 0
    # )
