import imp
import os
import torch
import sys
from packaging import version

torch_version = version.parse(torch.__version__)

if torch_version == version.parse("1.4.0"):
    lib_dir = os.path.dirname(__file__)
    _, path, _ = imp.find_module("_C", [lib_dir])
    torch.ops.load_library(path)
    import brevitas.function.ops_ste_n
    sys.modules['brevitas.function.ops_ste'] = brevitas.function.ops_ste_n
elif torch_version < version.parse("1.4.0"):
    import brevitas.function.ops_ste_o
    sys.modules['brevitas.function.ops_ste'] = brevitas.function.ops_ste_o
