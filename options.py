import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.opt = None

    def initialize(self):
        # Optimized for PyCharm Editor

        # glob
        self.parser.add_argument('--folder', type=str, default=None, help='a target folder directory')
        # folders (methods) glob
        self.parser.add_argument('--root', type=str, default=None, help='a root directory of target folders')

        # file extension
        self.parser.add_argument('--ext', type=str, default=['png'], nargs='+', help='extension of target files')
        # metrics
        self.parser.add_argument('--metric', type=str, default=None, nargs='+',
                                 help='piqe selection for evaluation',
                                 choices=['BRISQUE', 'NIQE', 'PIQE', 'METAIQA', 'RANKIQA'])
        self.parser.add_argument('--gpu', type=str2bool, default=False, help='Permission to use GPU for DeepIQA')

        # print settings
        self.parser.add_argument('--print_n', type=str2bool, default=True, help='print <n>th process')
        self.parser.add_argument('--verbose', type=str2bool, default=False, help='print detail process')
        self.parser.add_argument('--saveprint', type=str2bool, default=True, help='print save directory')

        # save settings
        self.parser.add_argument('--record', type=str, default='./result.txt', help='destination directory of record')
        self.parser.add_argument('--each_record', type=str2bool, default=False, help='record score of each image or not.')

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt
