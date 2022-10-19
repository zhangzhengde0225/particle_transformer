
from sympy import im


import os, sys
import argparse
from hai_api import hai


def eval(args):

    model = hai.hub.load(args.name)
    config = model.config
    config.source = args.source
    print(config)
    model.evaluate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default='particle_transformer', help='model name')
    parser.add_argument('-s', '--source', type=str, default='JetClass', help='input source, i.e. dataset name, supported: JetClass, TopLandscape, QuarkGluon')
    parser.add_argument('-f', '--feature_type', type=str, default='full', help='input feature type: kin, kinpid, full')

    args = parser.parse_args()
    
    eval(args)
