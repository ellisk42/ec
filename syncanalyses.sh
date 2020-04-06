#!/bin/bash

rsync -ravz /bluejay/home/dc/analysis/ /data1/code/python/ec/analysis
rsync -ravz /bluejay/home/dc/experimentOutputs/ /data1/code/python/ec/experimentOutputs
rsync -ravz /bluejay/home/dc/jobs/ /data1/code/python/ec/jobs
rsync -ravz /bluejay/home/dc/container.img /data1/code/python/ec

