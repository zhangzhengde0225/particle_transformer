"""
Patricle Transformer HAI API
"""

import os, sys
from pathlib import Path
import copy
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
        gpus = '""' if cfg.device == 'cpu' else cfg.device

        self.check_dataset_env(dataset)  # 搜索数据集环境
        cwd = os.getcwd()
        work_dir = f'{pydir.parent}/src'
        os.chdir(work_dir)


        code = f'bash {work_dir}/train_{dataset}.sh'  \
                f' {self.model_name}' \
                f' {feature_type}' \
                f' --gpus {gpus}'
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

    def check_dataset_env(self, dataset):
        """
        ：param dataset: 数据集名称， JetClass, JetClass-mini, etc.
        数据集环境写在src/env.sh里，需要同时几个搜索路径
        """

        search_dirs = [
            f"{Path.home()}/datasets",
            "/hepsfs/user/zdzhang/hai_datasets",
            "/home/zzd/datasets/hai_datasets",
        ]
        # 找到就重写
        new_dir = None
        for dirr in search_dirs:
            if not os.path.exists(dirr):
                continue
            files = os.listdir(dirr)
            if dataset in files:
                new_dir = dirr
                break
        if new_dir is not None:
            # 重写src/env.sh
            envfile = f'{pydir.parent}/src/env.sh'
            datadir = f'{new_dir}/{dataset}'
            lower_dataset = copy.copy(dataset).replace('-', '_')
            datapath = f'DATADIR_{lower_dataset}={datadir}'
            with open(envfile) as f:
                lines = f.readlines()
            with open(envfile, 'w') as f:
                for l in lines:
                    if f'DATADIR_{lower_dataset}' in l:
                        l = f'export {datapath}\n'
                    f.write(l)
            print(f'Updated dataset path in {envfile} to "{datapath}".')

