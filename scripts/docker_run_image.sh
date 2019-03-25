#!/bin/sh
docker run -it -p 6006:6006 -v /Users/lj/Projects/mlinseconds:/mlinseconds -w /mlinseconds --rm pytorch
