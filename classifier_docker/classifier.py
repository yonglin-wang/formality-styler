#!/usr/bin/env python3
# Author: Yonglin Wang
# Date: 2021/5/10 12:23 PM

import argparse
import re
import json
import requests

import web
import fasttext

APP = web.auto_application()
RM_PORT = 8081  # use the same number in $ python classifier.py 8081
HOST_PORT = 2500
LABEL_PATT = re.compile(r"__label__(\S+)")
model_path = "1gram_ft.ftz"

class classify(APP.page):
    def __init__(self):
        self.model = fasttext.load_model(model_path)
        print(f"{__name__} now up and listening!")

    def predict(self, s: str):
        label, score = self.model.predict(s)
        label = LABEL_PATT.match(label[0]).group(1)
        score = round(score[0], 3)
        return label, score

    def POST(self):
        raw_data = web.data()
        print(raw_data)

        data = json.loads(raw_data)
        text = data["text"]
        label, score = self.predict(text)
        response = {"label": label,
                    "score": score}

        return json.dumps(response)



# def main():
#     # command line parser
#     # noinspection PyTypeChecker
#     parser = argparse.ArgumentParser(prog="Name of Program",
#                                      description="Program Description",
#                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument("positional",
#                         type=int,
#                         help="a positional argument")
#     parser.add_argument('--float',
#                         type=float,
#                         default=0.5,
#                         help='optional float with default of 0.5')
#     parser.add_argument("-o", "--optional_argument",
#                         type=str,
#                         default=None,
#                         help="optional argument; shorthand o")
#     parser.add_argument("-t", "--now_true",
#                         action="store_true",
#                         help="boolean argument, stores true if specified, false otherwise; shorthand t")
#
#     args = parser.parse_args()
#

if __name__ == "__main__":
    APP.run()