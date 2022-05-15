#!/bin/bash

set -xeu

pytest \
  --tx 4*popen//python=python \
  --color yes \
  --cov saliency_metrics \
  --cov-report term-missing \
  --cov-report xml \
  -vvv \
  tests
