#!/bin/bash

liblinear-2.45/train -c 4 -e 0.1 -v 5 train.svm tweets.model
liblinear-2.45/train -c 4 -e 0.1 train.svm tweets.model
liblinear-2.45/predict test.svm tweets.model out.txt
