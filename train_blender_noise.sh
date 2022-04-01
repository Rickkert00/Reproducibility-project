#!/bin/bash
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Script for training on the Blender dataset.

SCENE=lego
EXPERIMENT=debug
TRAIN_DIR=nerf_results/$EXPERIMENT/lego_noise
DATA_DIR=data/nerf_raw_noise/$SCENE

#rm $TRAIN_DIR/*
python3 -m train \
  --data_dir=$DATA_DIR \
  --train_dir=$TRAIN_DIR \
  --gin_file=configs/rawNERF.gin \
  --gin_param="Config.batch_size = 1024"
  --logtostderr
