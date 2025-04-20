#!/bin/bash

#cd ../..

# custom config
DATA="path-to-data/"
TRAINER=TLAC

GEMINI_MODEL_ID_1="models/gemini-1.5-flash-latest" # For SLAC and first step of TLAC
GEMINI_MODEL_ID_2="models/gemini-1.5-flash-latest" # For second step of TLAC

GEMINI_API_KEY="" # Paste Your Gemini API key here


DATASET=$1
MODEL=$2

CFG=config

SUB=new


python test.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--model ${MODEL} \
--eval-only \
--gemini_model_1 ${GEMINI_MODEL_ID_1} \
--gemini_model_2 ${GEMINI_MODEL_ID_2} \
--gemini_api ${GEMINI_API_KEY} \
DATASET.SUBSAMPLE_CLASSES ${SUB}

