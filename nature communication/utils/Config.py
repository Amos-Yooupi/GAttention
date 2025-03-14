import json


class Config(object):
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            data = json.load(file)

        for key, value in data.items():
            setattr(self, key, value)