import os, sys, time, torch

from add2.test_add2 import test_add2
from reduce.test_reduce import test_reduce
from softmax.test_softmax import test_softmax
from self_attn.test_self_attn import test_self_attn
# 遍历测试目录运行所有测试
if __name__ == "__main__":
  root = os.getcwd()
  torch.ops.load_library(root + '/build/libCudaLab.so')
  test_add2()
  test_reduce()
  test_softmax()
  test_self_attn()
