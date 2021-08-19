#!/usr/bin/env python3
# Author: Yonglin Wang
# Date: 2021/3/17 10:17 PM

import argparse
import os
import random
from collections import Counter, defaultdict
import re

# modify these paths and run the script.
orig_paths = ["data/Entertainment_Music", "data/Family_Relationships"]      # path to the two domains in original corpus
OUT_PATH = "data/all"       # output path


# variables below should normally remain unchanged...
TRAIN = "train"
TUNE = "tune"
TEST = "test"

FORMAL = "formal"
INFORMAL = "informal"
LABELS = [FORMAL, INFORMAL]

# FORMAL_ENTRIES = 2101
# INFORMAL_ENTRIES = 2748

LABELLED_FORMAT = "__label__{} {}"     # first label, then content
LABEL_PATT = re.compile(r"^__label__(\S+) ([^\n]+)")


def combine_files(split_name, orig_paths, output_dir, from_informal=True):
    """
    combine domain data of a given split (i.e. train, test, or tune) into one file saved under output directory
    :param split_name: train, test, or tune
    :param orig_paths:
    :param output_dir:
    :param from_informal: whether from informal to formal for test and tune direction instead of formal to informal
    :return: path to fasttext data
    """
    split_out_file = os.path.join(output_dir, split_name)

    with open(split_out_file, "a+") as out_f:
        if split_name == "train":
            for orig_path in orig_paths:    # e.g. data/Family_Relationships
                for label in LABELS:
                    # e.g. data/Family_Relationships/train.informal
                    label_out = os.path.join(output_dir, split_name + f".{label}")
                    with open(os.path.join(orig_path, split_name, label), "r") as orig_file, \
                            open(label_out, "a+") as label_out_f:
                            for line in orig_file:
                                out_f.write(LABELLED_FORMAT.format(label, line))
                                label_out_f.write(line)
        else:
            # for test and tune, use formal and informal.ref0
            for orig_path in orig_paths:    # e.g. data/Family_Relationships
                split_path = os.path.join(orig_path, split_name)     # e.g. data/Family_Relationships/tune/
                with open(os.path.join(split_path, "formal")) as src_text, \
                     open(os.path.join(split_path, "informal.ref0")) as tgt_text, \
                     open(os.path.join(output_dir, split_name + ".formal"), "a+") as src_out, \
                     open(os.path.join(output_dir, split_name + ".informal"), "a+") as tgt_out:
                    # write src tgt pair to their respective files
                    for src, tgt in zip(src_text, tgt_text):
                        src_out.write(src)
                        tgt_out.write(tgt)
                        out_f.write(LABELLED_FORMAT.format("formal", src))
                        out_f.write(LABELLED_FORMAT.format("informal", tgt))

    shuffle_lines(split_out_file)


def shuffle_lines(file_name) -> None:
    """shuffle lines of data in a given file"""
    with open(file_name, "r") as f:
        lines = f.readlines()
        random.shuffle(lines)

    with open(file_name, "w") as f:
        f.writelines(lines)

def count_labels(filename):
    """for testing purpose, count occurences of __label__ in fasttext data file"""
    print(f"Stats for {filename}")
    label_stats = defaultdict(lambda: {"entry": 0, "tokens": 0, "characters": 0})

    with open(filename, "r") as f:
        for line in f:
            label = LABEL_PATT.match(line).group(1)
            sent = LABEL_PATT.match(line).group(2)

            label_stats[label]["entry"] += 1
            label_stats[label]["tokens"] += len(sent.split())
            label_stats[label]["characters"] += len(sent)

    total_ents = sum([d['entry'] for d in label_stats.values()])

    print(f"Total Sample Size: {total_ents}")
    for label, stats in label_stats.items():
        print(f"{label}: \n"
              f"Number of Samples: {stats['entry']} ({stats['entry']/total_ents:.5%})\n"
              f"Average Token/sent: {stats['tokens']/stats['entry']}\n"
              f"Average Char/sent: {stats['characters']/stats['entry']}")


def combine_test(dir1, dir2, outdir) -> None:
    """This function is used to combine .ref* files across domains for seq2seq model training."""
    # Not sure if this contains bugs. See README for expected behavior.
    dir1_files = set(os.listdir(dir1))
    dir2_files = set(os.listdir(dir2))
    os.makedirs(outdir, exist_ok=True)
    assert len(dir2_files) == 10 and len(dir1_files) == 10, "5 formal, 5 informal files expected"
    for file in dir1_files:
        print(f"now joining {file} in {dir1} and then {dir2}")
        with open(os.path.join(dir1, file), "r") as fin1, open(os.path.join(dir2, file), "r") as fin2, \
            open(os.path.join(outdir, file), "w") as fout:
            fout.writelines(fin1.readlines())
            fout.writelines(fin2.readlines())
    print("Done!")

    for file in os.listdir(outdir):
        lines = len(open(os.path.join(outdir, file), "r").readlines())
        print(f"{os.path.join(outdir, file)}: {lines} lines")



if __name__ == "__main__":
    [combine_files(split, orig_paths, OUT_PATH) for split in (TRAIN, TUNE, TEST)]
    
    # read and validate output files
    for root, dirs, files in os.walk(OUT_PATH):
        for file in files:
            file_name = os.path.join(root, file)
            print(f"total number of lines in {file_name}: ")
            if not file_name.endswith(tuple(LABELS)):
                count_labels(file_name)
            else:
                print(sum(1 for line in open(file_name)))
    
            print("*==*" * 20)

    # # ## Combine test set directory
    # combine_test("data/Entertainment_Music/test",
    #              "data/Family_Relationships/test",
    #              "data/combined_test")

