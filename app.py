#!/usr/bin/env python3
# Author: Yonglin Wang
# Date: 2021/5/2 3:38 PM

import argparse
from time import time

from flask import Flask, render_template, request

from styler import Generator
import consts as C

begin = time()
app = Flask(__name__)

# for debugging, informal command:
# TODO if I hit "rewrite" button too fast, the program breaks
# fairseq-interactive data-bin/informal-formal --path fairseq_results/informal-formal/checkpoint_best.pt --beam 5 --source-lang informal --target-lang formal --bpe subword_nmt --bpe-codes fairseq_data/bpe_output/bpe_code --tokenizer moses --moses-target-lang en  --remove-bpe
# formal-informal generator
for_inf_gen = Generator(C.FORMAL)
# informal-formal generator
inf_for_gen = Generator(C.INFORMAL)

# vars to be used across pages
direction_choice, src_text, asciifolding = "", C.DEFAULT_QUERY, False


# home page
@app.route("/", methods=["POST", "GET"])
def home():
    return render_template("home.html",
                           default_query=src_text,
                           request_too_frequent=False)

@app.route("/results", methods=["POST"])
def results():
    global src_text, direction_choice, asciifolding
    src_text = request.form["source_text"]
    direction_choice = request.form["direction_option"]
    asciifolding = request.form.get("asciifolding") is not None

    if asciifolding:
        src_text = Generator.clean_input(src_text)

    try:
        if direction_choice == "from_formal":
            output = for_inf_gen.get_translation(src_text)
        elif direction_choice == "from_informal":
            output = inf_for_gen.get_translation(src_text)
        elif direction_choice == "auto":
            output = "not implemented yet!"
        else:
            raise NotImplementedError
    except ValueError:
        # if request too often, pexpect will break; use ValueError to catch this
        src_text = C.DEFAULT_QUERY
        return render_template("home.html",
                               default_query=src_text,
                               request_too_frequent=True)

    return render_template("results.html",
                           default_query=src_text,
                           output_text=output.detokenized,
                           direction_choice=direction_choice,
                           asciifolding=asciifolding)

if __name__ == "__main__":
    # command line parser
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(prog="Formality Styler Program",
                                     description="Program to run Flask App for the formality styler.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--port",
                        type=int,
                        default=5000,
                        help="Port to start app at. Default: 5000")
    args = parser.parse_args()

    # start app
    print(f"App took {time() - begin} seconds to start.")
    app.run(debug=True, port=args.port)
