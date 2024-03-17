import os
import sys
import argparse
import numpy as np
import onnx_graphsurgeon as gs
import onnx
import configparser

def ArgsParse():
    # 创建一个ArgumentParser对象
    parser = argparse.ArgumentParser(description='convert decoder onnx weight 2 bin')
    parser.add_argument('-onnx_path', type=str, default='./model/model.onnx')
    parser.add_argument('-bin_path', type=str, default='./model/bin/')
    parser.add_argument('-fp16', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = ArgsParse()
    graph = gs.import_onnx(onnx.load(opt.onnx_path))
    graph.fold_constants().cleanup()

    onnx.save(gs.export_onnx(graph), "./model/model_fold.onnx")
    weights = {} 
    for node in graph.nodes:
        for input in node.inputs:
            if "helper" in input.name:
                continue
            print(input)
            if hasattr(input, "values"):
                print("Export {}".format(input.name))
                weights[input.name] = input.values
                
    conf = configparser.ConfigParser()
    conf.add_section("ernie")            
    
    with open(os.path.join(opt.bin_path, "config.ini"), 'w') as fid:
        if opt.fp16:
            print("Extract weights in FP16 mode")
            conf.set("ernie", "weight_data_type", "fp16")
            npDataType = np.float16
        else:
            print("Extract weights in FP32 mode")
            conf.set("ernie", "weight_data_type", "fp32")
            npDataType = np.float32
        conf.write(fid)
    
    for name,value in weights.items():
        saved_path = os.path.join(opt.bin_path, name+".bin")
        print(name, value.shape)
        value.astype(npDataType).tofile(saved_path)
    print("Succeed extracting weights of Ernie!")

    
    