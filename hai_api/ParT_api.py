"""
Patricle Transformer HAI API
"""

import os, sys
from pathlib import Path
pydir = Path(os.path.dirname(os.path.realpath(__file__)))
import hai
from hai import AbstractModule, AbstractInput, AbstractOutput, AbstractQue
from hai import MODULES, SCRIPTS, IOS, Config


@MODULES.register()
class ParT(AbstractModule):
    name = 'Particle_Transformer'  # Specify the name of your model explicitly
    description = '2022 SOTA Jet tagging algorithm'  # Specify the description of your model

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        Please set the default configuration of your model here
        """
        self._train_cfg = f'{pydir}/config.py'
        # The default config can be a dict, json, yaml, a python config file or a config object from argparse
        # and it will be converted to a Config object automatically by the framework
        self.model_name = 'ParT'
    
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
        work_dir = f'{pydir.parent}/src'
        os.chdir(work_dir)

        code = f'bash {work_dir}/train_{dataset}.sh'  \
                f' {self.model_name}' \
                f' {feature_type}' 
        os.system(code)

        os.chdir(cwd)

        # raise NotImplementedError(f'{self.name}.train() is not implemented, plese check the api: "{self.__module__}"')

    def infer(self, *args, **kwargs):
        """
        Please implement your model inference here
        """
        raise NotImplementedError(f'{self.name}.infer() is not implemented, plese check the api: "{self.__module__}"')

    def evaluate(self, *args, **kwargs):
        """
        Please implement your model evaluation here
        """
        cfg = self.cfg
        dataset = cfg.source
        feature_type = cfg.feature_type

        cwd = os.getcwd()
        work_dir = f'{pydir.parent}/src'
        os.chdir(work_dir)

        code = f'bash {work_dir}/train_{dataset}.sh'  \
                f' {self.model_name}' \
                f' {feature_type}' 
        os.system(code)

        os.chdir(cwd)


        raise NotImplementedError(f'{self.name}.evaluate() is not implemented, plese check the api: "{self.__module__}"')