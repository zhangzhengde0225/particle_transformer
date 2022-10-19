"""
train model via HAI
"""
import os, sys
import argparse
import json
import numpy as np
from hai_api import hai


def train(args):

    models = hai.hub.list()
    print('models')
    model = hai.hub.load(args.name)
    config = model.config
    config.source = args.source
    config.device = args.device
    print(config)
    model.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default='particle_transformer', help='model name')
    parser.add_argument('-s', '--source', type=str, default='JetClass-mini', help='input source, i.e. dataset name, supported: JetClass, TopLandscape, QuarkGluon')
    parser.add_argument('-f', '--feature_type', type=str, default='full', help='input feature type: kin, kinpid, full')
    parser.add_argument('-d', '--device', type=str, default='0', help='Device to use for training, e.g. cpu or 0 or 0,1,2,3 for GPU')

    args = parser.parse_args()
    
    train(args)
 
