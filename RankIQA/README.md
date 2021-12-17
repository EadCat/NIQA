# Pytorch-TestRankIQA

https://github.com/YunanZhu/Pytorch-TestRankIQA (2021.11.06 Transplanted)

RankIQA was proposed in a ICCV2017 paper by [Liu X](https://github.com/xialeiliu). You can get this paper from [arXiv](https://arxiv.org/abs/1707.08347) or [ICCV 2017 open access](https://openaccess.thecvf.com/content_iccv_2017/html/Liu_RankIQA_Learning_From_ICCV_2017_paper.html).

This repo contains [RankIQA](https://github.com/xialeiliu/RankIQA) model files in Pytorch.  
You can use it to test RankIQA on TID2013 and LIVE dataset in Pytorch.  
If you just want to have a quick and simple comparison with RankIQA on your own test set, this repo is suited for you.

## News
- We have released our recent work [RecycleD](https://github.com/YunanZhu/RecycleD). It has been accepted to *[ACM MM 2021 brave new ideas](https://2021.acmmm.org/brave-new-ideas-proposals)*.  
  It is an opinion-unaware non-reference IQA method which is based on the pre-trained discriminator of WGAN.  
  It may interest you if you are trying to use or study RankIQA.  

## Prerequisites
* Win10 (Not tested on Ubuntu yet)
* Python 3.6
* Numpy 1.19.1
* Pytorch 1.2
* Opencv 4.5

The above versions are not mandatory, just because I ran the code in such an environment.

## Getting Started
```
python main.py --test_set "ur_path/TID2013/" --model_file "./pre-trained/Rank_tid2013.caffemodel.pt" --test_file "./data/ft_tid2013_test.txt" --res_file "./result.csv"
python main.py --test_set "ur_path/TID2013/" --model_file "./pre-trained/FT_tid2013.caffemodel.pt" --test_file "./data/ft_tid2013_test.txt" --res_file "./result.csv"

python main.py --test_set "ur_path/LIVE2/" --model_file "./pre-trained/Rank_live.caffemodel.pt" --test_file "./data/ft_live_test.txt" --res_file "./result.csv"
python main.py --test_set "ur_path/LIVE2/" --model_file "./pre-trained/FT_live.caffemodel.pt" --test_file "./data/ft_live_test.txt" --res_file "./result.csv"
```
Note: ```test_set``` is the dataset folder, ```model_file``` is the pre-trained model file, ```test_file``` is the txt file which contains MOS and image filenames (see [here](data/)), ```res_file``` is the csv file to save the test results.

## About the pre-trained model files
I use [caffemodel2pytorch](https://github.com/vadimkantorov/caffemodel2pytorch) to transform the [Caffe](http://caffe.berkeleyvision.org/) model file to Pytorch format.
You can find the pre-trained Caffe model files of RankIQA in [here](https://github.com/xialeiliu/RankIQA/tree/master/pre-trained).

***You can also download the Pytorch model files transformed by myself from:***

- ***[Baidu disk](https://pan.baidu.com/s/1HjYFypg-RWE-W-TvNQ-02A), and the password is ```riqa```.***
- ***[Google drive](https://drive.google.com/drive/folders/1OQ0IQrWoricMhaIyfwqsJVlYpXHKPP1z).***

I have used these pre-trained model files to test the performance on LIVE2 and TID2013.  
The test results are recorded in [test results.xlsx](test%20results.xlsx).

## Tips
***I cannot guarantee the correctness of the pre-trained Pytorch model files and the test results.***

I just tried to reproduce the results showed in the [paper](https://openaccess.thecvf.com/content_iccv_2017/html/Liu_RankIQA_Learning_From_ICCV_2017_paper.html),
and you can see the reproduced results on [TID2013](https://github.com/YunanZhu/Pytorch-TestRankIQA/blob/main/results%20of%20RankIQA%20on%20LIVE.xlsx) and [LIVE](https://github.com/YunanZhu/Pytorch-TestRankIQA/blob/main/results%20of%20RankIQA%20on%20TID2013.xlsx).

If you are familiar with Pytorch, you can modify the code and test RankIQA on other datasets.  
I didn't write the training code.
