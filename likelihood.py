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
            # print "Probs shape: "+str(probs[1].shape)
            # print "Chars shape: "+str(len(chars))
            # print "All chars: "  +str(chars)
            print "==============================================="
            print "Input:Prob.\t1st Pred\t2nd Pred\t3rd Pred"
            print "'" + charlist[0]  + "'"
            for c, p in zip(charlist[1:], probs[:-1]):
                # convert the p, which is (1,c) in shape, to a 1-dim arraw
                p = p.reshape(p.shape[1])
                # find the index of c in chars,
                # see http://stackoverflow.com/questions/9542738/python-find-in-list
                i = chars.index(c)
                # get the prob of c and its log10.
                prob = p[i]
                logProb = -math.log10(prob)
                # get the 3 highest probs. Print as {'char':prob}
                # first argsort probs to get the index of sorted probs.
                order = p.argsort()
                top3= order[::-1][:3]
                # print top3
                # print p[top3]
                # print type(chars)
                chlist = np.array(list(chars))
                # print chlist[top3]
                # print zip(chlist[top3], p[top3])

                top3output = "\t".join("'"+str(ci)+"'"+": %.4f"%pi for ci, pi in zip(chlist[top3], p[top3]))
                firstChoice= "*" if c==chlist[top3[0]] else ""
                # print
#                print "'" + c +"':" + "%.5f" %prob + "\t"+ "%.5f" %logProb + top3output
                print "'" + c +"':" + "%.5f" %prob + firstChoice +"\t"+ top3output
                sumLogProb += logProb
                #print c + " (" + str(chars.index(c))+ "): " + str(prob)
                # print (p)
                # print ""
            print "==============================================="
            print "Log Joint Prob= -" + str(sumLogProb)

if __name__ == '__main__':
    main()
