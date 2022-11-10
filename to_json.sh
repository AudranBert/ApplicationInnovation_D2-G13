#!/bin/bash

pip3 install yq
cat dataset/dev.xml | xq . > dataset/dev.json
cat dataset/train.xml | xq . > dataset/train.json
