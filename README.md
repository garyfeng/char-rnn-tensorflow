# Log

## 2015-12-29: adding likelihood calculation

After you train a RNN, you can use the following to calc the prob. of a
text string given the RNN:

```
python likelihood.py --save_dir "data/keystrokeBanAds/save_rnn" --text "The adves_rtising industy and spend _s over"
```

This prints out

 - some diagnostic info about the character map, etc.
 - the input letter string, prob(letter|model,priorText), and (negative) log prob.
 - the log joint probability, i.e., log10(p(string|model))

```
Probs shape: (1, 96)
Chars shape: 96
All chars: (' ', '_', 'e', 't', 'a', 'n', 'o', 'i', 's', 'h', 'r', '=', 'd', 'l', 'u', 'c', 'm', 'y', 'g', 'w', 'b', 'f', 'v', '.', 'p', '\n', 'k', '\\', ',', '1', 'I', 'T', '0', '2', 'A', '3', '4', 'S', "'", '5', '8', 'j', '6', 'W', '7', '9', 'U', 'M', 'x', 'B', 'D', 'E', 'C', 'O', 'N', 'H', '?', '"', ';', 'F', 'z', 'P', 'Y', 'q', 'L', 'V', '!', 'R', '/', 'K', 'G', '-', 'J', ':', '>', '[', '$', ')', '<', ']', '(', '%', 'X', 'Q', '&', '+', '@', 'Z', '*', '`', '|', '#', '}', '{', '^', '~')
===============================================
Char	Prob.	Negative Log Prob
'T'
'h'	0.23099	0.63640
'e'	0.52314	0.28138
' '	0.31403	0.50303
'a'	0.16626	0.77921
'd'	0.51681	0.28667
'v'	0.63606	0.19650
'e'	0.93321	0.03002
's'	0.00122	2.91257
'_'	0.20101	0.69678
'r'	0.11417	0.94246
't'	0.14667	0.83367
'i'	0.87135	0.05981
's'	0.92512	0.03380
'i'	0.51324	0.28968
'n'	0.94099	0.02642
'g'	0.93771	0.02793
' '	0.85444	0.06832
'i'	0.09485	1.02296
'n'	0.21915	0.65926
'd'	0.26248	0.58091
'u'	0.88224	0.05441
's'	0.88013	0.05545
't'	0.96248	0.01661
'y'	0.18878	0.72404
' '	0.66937	0.17433
'a'	0.09718	1.01241
'n'	0.54274	0.26541
'd'	0.91169	0.04015
' '	0.97999	0.00878
's'	0.08183	1.08707
'p'	0.02607	1.58391
'e'	0.79125	0.10169
'n'	0.90602	0.04286
'd'	0.75186	0.12387
' '	0.50463	0.29703
'_'	0.02740	1.56218
's'	0.03565	1.44800
' '	0.88345	0.05382
'o'	0.05838	1.23373
'v'	0.01786	1.74806
'e'	0.64174	0.19264
'r'	0.94630	0.02397
===============================================
Log Joint Prob= -22.7181954377
```

## 2015-12-28: Keystroke log representations

Our first pet project is to convert the old Keystroke logs to a format that
can be used as input for training. For this, we took the simplest approach, where
we encoded backspacing at the end of the text as `_` and ignored all editing
events *not* at the end of the text. The resulting log looks like this:

```
Sholud the people ban Marki ng____eting for h_children under twelve?\r\n     I dont think that they should bc people who have children can_______ Because marketing is not all bad.
```

Remember that each '_' character means backspacing (from the end of the line). So
'Marki ng____eting' means eating back 'i ng' and typing 'eting' to make
'marketing'.

With this representation we are able to let the RNN learn the dynamics of
editing, albeit a simple kind.

The initial try used LSTM, with default parameters from the `char-rnn-tensorflow`
code base. The result (reported elsewhere) is encouraging. 

## 2015-12-20: Adding

We want to calculate the prob. of an observed text given a model, let's
modify the `sample.py`. Instead of generating one char at a time and then select
the one with the highest probability, we calculate the prob for the actual char
from the text. Then we calc the overall prob. over the string (assuming
independence?).


# char-rnn-tensorflow
Multi-layer Recurrent Neural Networks (LSTM, RNN) for character-level language models in Python using Tensorflow.

Inspired from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn).

# Requirements
- [Tensorflow](http://www.tensorflow.org)

# Basic Usage
To train with default parameters on the tinyshakespeare corpus, run `python train.py`.

To sample from a checkpointed model, `python sample.py`.
# Roadmap
- Add explanatory comments
- Expose more command-line arguments
- Compare accuracy and performance with char-rnn
