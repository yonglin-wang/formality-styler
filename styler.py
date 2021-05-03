#!/usr/bin/env python3
# Author: Yonglin Wang
# Date: 2021/4/30 10:19 PM

import argparse

#!/usr/bin/env python3
# Author: Yonglin Wang
# Date: 2021/4/24 10:52 PM
# Put this script under project root!
# Initiates fairseq interactive mode and generates the translation

import re
import argparse
import typing

import pexpect
import attr
from fold_to_ascii import fold

import consts as C


S_PATTERN = re.compile(r"S-(\d+)\s+(.+)")
W_PATTERN = re.compile(r"W-\d+\s+(\d+\.\d+)\s+")
H_PATTERN = re.compile(r"H-\d+\s+-\d\.\d+\s+(.+)")
D_PATTERN = re.compile(r"D-\d+\s+-\d\.\d+\s+(.+)")
P_PATTERN = re.compile(r"P-\d+\s+(.+)")

@attr.s(auto_attribs=True)
class InteractiveCommand:
    """class for generating correct fairseq-interactive command"""
    src: str = attr.ib(validator=attr.validators.instance_of(str))
    tgt: str = attr.ib()
    beam: int = attr.ib(5)
    remove_bpe: bool = attr.ib(True)
    bpe_path: str = attr.ib(C.BPE_CODE_PATH)
    data_bin_path: str = attr.ib()
    model_path: str = attr.ib()

    @src.validator
    def _validate_source(self, attribute, value):
        if value not in (C.FORMAL, C.INFORMAL):
            raise ValueError(f"{attribute.name} must be either {C.FORMAL} or {C.INFORMAL}")

    @tgt.default
    def _infer_tgt(self):
        return C.INFORMAL if self.src == C.FORMAL else C.FORMAL

    @data_bin_path.default
    def _default_bin(self):
        return C.IF_DATABIN if self.src == C.INFORMAL else C.FI_DATABIN

    @model_path.default
    def _default_model_path(self):
        return C.IF_MODEL_PATH if self.src == C.INFORMAL else C.FI_MODEL_PATH

    def generate_command(self):
        return C.COMMAND_FORMAT.format(self.data_bin_path,
                                     self.model_path,
                                     self.beam,
                                     self.src, self.tgt,
                                     self.bpe_path,
                                     C.REMOVE_BPE if self.remove_bpe else "")


@attr.s(auto_attribs=True)
class Output:
    raw_src_text: str = attr.ib()
    token_src_text: str = attr.ib()
    wait_time: float = attr.ib()
    hypothesis: str = attr.ib()
    detokenized: str = attr.ib()
    session_id: int = attr.ib()
    token_prob: typing.List[float] = attr.ib()


class Generator:
    def __init__(self, src: str):
        self.src = src
        self.ic = InteractiveCommand(self.src)
        self.child = self._spawn_child()
        # first call is slower--run a placeholder translation, then the future method calls will be speedy
        self.get_translation(" ")

    def _spawn_child(self):
        """spawn new, input-ready child based on command tyepe"""
        child = pexpect.spawn(self.ic.generate_command(), encoding="utf8")
        # # prevent sendline too frequent: https://pexpect.readthedocs.io/en/stable/commonissues.html?highlight=%20delaybeforesend#timing-issue-with-send-and-sendline
        # self.child.delaybeforesend = 1
        # self.child.delayafterread = 0.2
        child.expect(C.INPUT_PROMPT)
        return child

    def _restart_child(self):
        """print debug info and reborn child, called when readline() did not behave as expected"""
        print("Bad stdout reading!")
        print(f"child.before: {self.child.before}")
        print(f"child.buffer: {self.child.buffer}")
        print("If this message shows up too many times, please manually jump to homepage.")
        self.child.terminate(force=True)
        self.child = self._spawn_child()

    @classmethod
    def clean_input(cls, input_str: str) -> str:
        return fold(input_str)

    def get_translation(self, input_str: str):
        # ### send source text to subprocess
        try:
            # if request too fast, might have OSError: [Errno 5] Input/output error
            self.child.sendline(input_str)
        except OSError as e:
            self._restart_child()
            raise ValueError("Connection failure! Please getting translate try again." + str(e))

        # ### read stdout record the translation output, always strip text to get rid of stdout "\r\n" at the end
        # line 1: original input
        ori_txt = self.child.readline().strip()
        try:
            # line 2: Source (tokenized)
            source_match = S_PATTERN.match(self.child.readline())
            session_id = source_match.group(1)
            src_txt = source_match.group(2).strip()

            # line 3: Wait time
            wait_time = W_PATTERN.match(self.child.readline()).group(1).strip()

            # line 4: Hypothesis from model
            hypo_txt = H_PATTERN.match(self.child.readline()).group(1).strip()

            # line 5: Detokenized hypothesis
            detok_txt = D_PATTERN.match(self.child.readline()).group(1).strip()

            # line 6: Position-wise token log probability
            prob_txt = P_PATTERN.match(self.child.readline()).group(1).strip()
        except AttributeError as e:
            # if match is None, AttributeError will occur. Catch and respawn child
            self._restart_child()
            raise ValueError("Buffer state mismatch! Please getting translate try again." + str(e))

        out = Output(ori_txt,
                     src_txt,
                     float(wait_time),
                     hypo_txt,
                     detok_txt,
                     int(session_id),
                     [float(prob) for prob in prob_txt.split()])

        return out

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


if __name__ == "__main__":
    gen1 = Generator(C.INFORMAL)
    # gen2 = Generator(C.FORMAL)