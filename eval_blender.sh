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

# Script for evaluating on the Blender dataset.

#SCENE=lego
#EXPERIMENT=debug
#TRAIN_DIR=/Users/barron/tmp/nerf_results/$EXPERIMENT/$SCENE # weights/checkpoint
#DATA_DIR=/Users/barron/data/nerf_synthetic/$SCENE #training


SCENE=lego
EXPERIMENT=debug
TRAIN_DIR1=nerf_results/saved/all_results/nerf_raw_noise_90_1024
DATA_DIR1=nerf_results/saved/all_data_sets/ #nerf_raw_noise_60_train


python3 -m eval \
  --data_dir=$DATA_DIR1 \
  --train_dir=$TRAIN_DIR1 \
  --chunk=3076 \
  --gin_file=configs/rawNERF.gin \
  --logtostderr

SCENE=lego
EXPERIMENT=debug
TRAIN_DIR2=nerf_results/saved/all_results/nerf_raw_noise_60_1024
DATA_DIR2=nerf_results/saved/all_data_sets/ #nerf_raw_noise_60_train


python3 -m eval \
  --data_dir=$DATA_DIR2 \
  --train_dir=$TRAIN_DIR2 \
  --chunk=3076 \
  --gin_file=configs/rawNERF.gin \
  --logtostderr

SCENE=lego
EXPERIMENT=debug
TRAIN_DIR3=nerf_results/saved/all_results/nerf_raw_noise_30_1024
DATA_DIR3=nerf_results/saved/all_data_sets/ #nerf_raw_noise_60_train


python3 -m eval \
  --data_dir=$DATA_DIR3 \
  --train_dir=$TRAIN_DIR3 \
  --chunk=3076 \
  --gin_file=configs/rawNERF.gin \
  --logtostderr

SCENE=lego
EXPERIMENT=debug
TRAIN_DIR4=nerf_results/saved/all_results/v2_nerf_raw_120_16384
DATA_DIR4=nerf_results/saved/all_data_sets/ #nerf_raw_noise_60_train


python3 -m eval \
  --data_dir=$DATA_DIR4 \
  --train_dir=$TRAIN_DIR4 \
  --chunk=3076 \
  --gin_file=configs/rawNERF.gin \
  --logtostderr
