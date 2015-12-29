# garyfeng
# modified from sample.py to calculate the -loglikelihood of a string
#   given the model.

import numpy as np
import tensorflow as tf

import argparse
import time
import os
import cPickle
import math

from utils import TextLoader
from model import Model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                       help='model directory to store checkpointed models')
    # parser.add_argument('-n', type=int, default=500,
    #                    help='number of characters to sample')
    parser.add_argument('--text', type=str, default='The ',
                       help='target text to evaluate')
    args = parser.parse_args()
    loglikelihood(args)

def loglikelihood(args):
    with open(os.path.join(args.save_dir, 'config.pkl')) as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl')) as f:
        chars, vocab = cPickle.load(f)
    model = Model(saved_args, True)
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            [charlist, probs] = model.loglikelihood(sess, chars, vocab, args.text)
            # print
            sumLogProb = 0.0
            print "Probs shape: "+str(probs[1].shape)
            print "Chars shape: "+str(len(chars))
            print "All chars: "  +str(chars)
            print "==============================================="
            print "Char\tProb.\tNegative Log Prob"
            print "'" + charlist[0]  + "'"
            for c, p in zip(charlist[1:], probs[:-1]):
                # find the index of c in chars,
                # see http://stackoverflow.com/questions/9542738/python-find-in-list
                prob = p[0][chars.index(c)]
                logProb = -math.log10(prob)
                print "'" + c +"'\t" + "%.5f" %prob + "\t"+ "%.5f" %logProb
                sumLogProb += logProb
                #print c + " (" + str(chars.index(c))+ "): " + str(prob)
                # print (p)
                # print ""
            print "==============================================="
            print "Log Joint Prob= -" + str(sumLogProb)

if __name__ == '__main__':
    main()
