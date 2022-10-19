"""
Particle-level Convolutional Neural Network (PCNN) HAI API
"""


import os, sys
from pathlib import Path
pydir = Path(os.path.dirname(os.path.realpath(__file__)))
import hai
from hai import AbstractModule, AbstractInput, AbstractOutput, AbstractQue
from hai import MODULES, SCRIPTS, IOS, Config


@MODULES.register()
class PCNN(AbstractModule):
    name = 'PCNN'  # Specify the name of your model explicitly
    description = '2019 Classic Jet tagging algorithm'  # Specify the description of your model

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        Please set the default configuration of your model here
        """
        self.default_cfg = f'{pydir}/config.py'
        # The default config can be a dict, json, yaml, a python config file or a config object from argparse
        # and it will be converted to a Config object automatically by the framework
        self.model_name = 'PCNN'
    
    def _init_model(self):
        """
        Please implement your model initialization here
        return: model
        """
        raise NotImplementedError(f'{self.name}._init_model() is not implemented, plese check the api: "{self.__module__}"')

    def train(self, *args, **kwargs):
        """
        Please implement your model training here
        """
        cfg = self.cfg
        dataset = cfg.source
        feature_type = cfg.feature_type

        cwd = os.getcwd()
        os.chdir(f'{pydir.parent}')

        code = f'bash {pydir.parent}/train_{dataset}.sh'  \
                f' {self.model_name}' \
                f' {feature_type}' 
        os.system(code)

        os.chdir(cwd)