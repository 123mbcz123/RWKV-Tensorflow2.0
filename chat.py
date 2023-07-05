########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, copy, types, gc, sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{current_path}/../rwkv_pip_package/src')

import numpy as np
from prompt_toolkit import prompt
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
np.set_printoptions(precision=4, suppress=True, linewidth=200)
args = types.SimpleNamespace()

print('\n\nChatRWKV v2 https://github.com/BlinkDL/ChatRWKV')

import torch


args.strategy = 'cpu fp32'

os.environ["RWKV_JIT_ON"] = '0' # '1' or '0', please use torch 1.13+ and benchmark speed
os.environ["RWKV_CUDA_ON"] = '0' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

args.MODEL_NAME = 'C:/Users/a1313/Desktop/RWKV-4-World-0.1B-v1-20230520-ctx4096'


if args.MODEL_NAME.endswith('/'): # for my own usage
    if 'rwkv-final.pth' in os.listdir(args.MODEL_NAME):
        args.MODEL_NAME = args.MODEL_NAME + 'rwkv-final.pth'
    else:
        latest_file = sorted([x for x in os.listdir(args.MODEL_NAME) if x.endswith('.pth')], key=lambda x: os.path.getctime(os.path.join(args.MODEL_NAME, x)))[-1]
        args.MODEL_NAME = args.MODEL_NAME + latest_file


from rwkv.model import RWKV


print(f'Loading model - {args.MODEL_NAME}')
model = RWKV(model=args.MODEL_NAME, strategy=args.strategy)
inputs_token = [i for i in range(192)]
outputs,states = model.forward(inputs_token,state=None)
