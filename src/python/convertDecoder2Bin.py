import os
import sys
import argparse
import numpy as np

def ArgsParse():
    # 创建一个ArgumentParser对象
    parser = argparse.ArgumentParser(description='convert decoder onnx weight 2 bin')
    parser.add_argument('--onnx_path', type=str, default='./model/model.opt.onnx')
    parser.add_argument('--bin_path', type=str, default='./model/bin')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = ArgsParse()
    print(opt.bin_path)
