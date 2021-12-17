import glob
import os.path
import cv2
from tqdm import tqdm

from brisque.brisque import brisque
from niqe.niqe import niqe
from piqe.piqe import piqe
from MetaIQA.model import MetaIQA
from RankIQA.model import RankIQA

import torch
from utils import csv_record


def metric_pass(*args, **kwargs):
    return 0


def bring_name(metric):
    try: return metric.__name__
    except: return metric.__class__.__name__


class NiqaEngine:
    def __init__(self, opt):
        self.opt = opt

        self.brisque_score = None
        self.niqe_score = None
        self.piqe_score = None
        self.metaiqa_score = None
        self.rankiqa_score = None
        # Build CNN under conditions
        if 'METAIQA' in self.opt.metric: self.metaIQA = MetaIQA(opt)
        if 'RANKIQA' in self.opt.metric: self.rankIQA = RankIQA(opt)

        self.method_idx = 1
        self.total_method = None

        self.metric_dict = {}

        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    def metric_brisque(self, imglist):
        self.metric_funct(brisque, imglist)

    def metric_niqe(self, imglist):
        self.metric_funct(niqe, imglist)

    def metric_piqe(self, imglist):
        self.metric_funct(piqe, imglist)

    def metric_metaiqa(self, imglist):
        self.metric_funct(self.metaIQA, imglist)

    def metric_rankiqa(self, imglist):
        self.metric_funct(self.rankIQA, imglist)

    def metric_choice(self) -> list:
        if self.opt.metric:
            # activate metrics under conditions
            metric_list = []
            if 'BRISQUE' in self.opt.metric:
                metric_list.append(self.metric_brisque)
            if 'NIQE' in self.opt.metric:
                metric_list.append(self.metric_niqe)
            if 'PIQE' in self.opt.metric:
                metric_list.append(self.metric_piqe)
            if 'METAIQA' in self.opt.metric:
                metric_list.append(self.metric_metaiqa)
            if 'RANKIQA' in self.opt.metric:
                metric_list.append(self.metric_rankiqa)
        else:
            metric_list = [metric_pass]
        return metric_list

    def glob_img(self, root):
        imglist = []
        for ext in self.opt.ext:
            target_ext = '*.' + ext
            imglist += sorted(glob.glob(os.path.join(root, target_ext)))
        return imglist

    def record_folder(self, imglist):
        log_dir = self.opt.record
        csv_record(self.metric_dict, log_dir, index=imglist)

    def method_list_up(self):
        if self.opt.folder is not None:
            print(f"1. {os.path.basename(self.opt.folder)}")
        elif self.opt.root is not None:
            folder_list = sorted(glob.glob(os.path.join(self.opt.root, '*')))
            for i, folder in enumerate(folder_list, 1):
                print(f"{i}. {os.path.basename(folder)}")
            print(f"total methods: {len(folder_list)}")

    def metric_funct(self, metric, imglist):
        total_len = len(imglist)
        assert total_len > 0, print(f"Input image list is empty.")
        total_score = 0.
        funct_name = bring_name(metric)
        each_score = []
        for img in tqdm(imglist, desc=f'{self.method_idx}/{self.total_method} {funct_name}: '):
            image = cv2.imread(img)
            score = metric(image)
            total_score += score

            if self.opt.each_record:
                if isinstance(score, torch.Tensor): score = score.item()
                each_score.append(score)

        total_score /= total_len  # average
        if isinstance(total_score, torch.Tensor): total_score = total_score.item()
        if self.opt.each_record:
            each_score.append(total_score)
            total_score = each_score
        self.metric_dict[funct_name] = total_score

    def measure(self, *args, **kwargs):
        metrics = self.metric_choice()  # list
        if self.opt.verbose: self.method_list_up()

        if self.opt.folder is not None:
            self.total_method = 1
            imglist = self.glob_img(self.opt.folder)
            for metric in metrics:
                metric(imglist)

            indices = ['Average']
            if self.opt.each_record:
                imglist = [os.path.basename(name) for name in imglist]
                indices = imglist + indices
            self.record_folder(imglist=indices)

        elif self.opt.root is not None:
            folder_list = sorted(filter(os.path.isdir, glob.glob(os.path.join(self.opt.root, '*'))))  # a list of folders
            self.total_method = len(folder_list)
            for folder in folder_list:
                print(f'[{self.method_idx}/{self.total_method}] {os.path.basename(folder)} start')
                imglist = self.glob_img(folder)
                for metric in metrics:
                    metric(imglist)

                indices = [f'{os.path.basename(folder)} Average']
                if self.opt.each_record:
                    imglist = [os.path.basename(name) for name in imglist]
                    indices = imglist + indices
                self.record_folder(imglist=indices)
                self.method_idx += 1

    def __call__(self, *args, **kwargs):
        return self.measure(*args, **kwargs)







