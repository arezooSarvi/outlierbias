'''
Created on 25 May 2021

@author: aliv
'''
from data_util import *
from attention import *


from absl import app
from absl import flags
import os
import sys

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('dataset', 'yahoo', 'dataset pickle address')
    flags.DEFINE_string('datasets_info', 'datasets_info.json', 'datasets info json file address')
    flags.DEFINE_string('epochs', '50', 'num of epochs')
    flags.DEFINE_string('results', '/ivi/ilps/personal/avardas/_data/outlier/', 'results address')
    flags.DEFINE_string('config', 'allrank/config.json', 'results address')
    flags.DEFINE_string('learning_rate', '0.001', 'learning rate')
    flags.DEFINE_string('jobid', '777', 'job ID')
    flags.DEFINE_string('sessions', '30', '')
    flags.DEFINE_string('rseed', '7', 'random seed.')
    flags.DEFINE_string('correction', 'affine', '"IPS", "affine", or "naive"')
    flags.DEFINE_boolean('verbose', False, '')
    flags.DEFINE_boolean('bernoulli', False, '')
    
    
  
  
def main(args):
    epochs = eval(FLAGS.epochs)
    learning_rate = eval(FLAGS.learning_rate)
    rseed = eval(FLAGS.rseed)

    dataset = load_dataset(FLAGS.dataset, FLAGS.datasets_info, int(FLAGS.sessions))
    with open(FLAGS.config) as f:
        net_config = json.load(f)
    
    net = cltr(jobid = FLAGS.jobid, dataset_name = FLAGS.dataset, correction_method = FLAGS.correction,
               net_config = net_config, dataset = dataset, 
               epochs = epochs, learning_rate = learning_rate, rseed = rseed, 
               bernoulli = FLAGS.bernoulli, 
               is_rbem = False,
               results_file = FLAGS.results, verbose = FLAGS.verbose)


if __name__ == '__main__':
    app.run(main)
