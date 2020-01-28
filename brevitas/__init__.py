import imp
import os
import torch

lib_dir = os.path.dirname(__file__)
_, path, _ = imp.find_module("_C", [lib_dir])
torch.ops.load_library(path)