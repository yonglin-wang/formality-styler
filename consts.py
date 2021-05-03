#!/usr/bin/env python3
# Author: Yonglin Wang
# Date: 2021/5/2 4:26 PM
# Constants for this project

# Styler.py
# paths
BPE_CODE_PATH = "data-bin/bpe_code"
# informal-formal static
IF_DATABIN = "data-bin/informal-formal"
IF_MODEL_PATH = "fairseq_results/informal-formal/checkpoint_best.pt"
# formal-informal static
FI_DATABIN = "data-bin/formal-informal"
FI_MODEL_PATH = "fairseq_results/formal-informal/checkpoint_best.pt"

COMMAND_FORMAT = "fairseq-interactive {} " \
             "--path {} " \
             "--beam {} " \
             "--source-lang {} --target-lang {} " \
             "--bpe subword_nmt --bpe-codes {} " \
             "--tokenizer moses --moses-target-lang en  " \
             "{}"

INFORMAL = "informal"
FORMAL = "formal"
REMOVE_BPE = "--remove-bpe"

# this is the last line of prompt after running the fairseq-interactive command!
INPUT_PROMPT = "Type the input sentence and press return:\r\n"

# app.py constant
DEFAULT_QUERY = "Hello, world!"

