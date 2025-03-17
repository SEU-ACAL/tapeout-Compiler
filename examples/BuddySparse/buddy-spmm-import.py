import os
from pathlib import Path
import torch
import torch._inductor.lowering
import numpy as np

from torch._inductor.decomposition import decompositions as inductor_decomp
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.graph import GraphDriver

# def sparse_matmul(x, y):
#     result = torch.mm(x, y)
#     result = torch.mm(y, result)
#     return result

# def foo(x, y, z):
#     return sparse_matmul(x, y)
#     # return torch.matmul(z, y)
#     # return torch.matmul(x, y)

# torch.library.define("mylib::sparse_matmul", "(Tensor x, Tensor y) -> Tensor")
# def sparse_matmul_impl(x, y):
#     result = torch.mm(x, y)
#     return result # torch.mm(y, result)
# torch.library.impl("mylib::sparse_matmul", "default", sparse_matmul_impl)

def foo(x, y, z):
    # return torch.ops.mylib.sparse_matmul(z, y)
    return torch.mm(x, y)


in1 = torch.randn(3, 4, dtype=torch.float32)
in2 = torch.randn(4, 3, dtype=torch.float32)

in3 = torch.tensor([[0, 0, 1, 0], 
                    [1, 2, 0, 0], 
                    [0, 0, 0, 0]], dtype=torch.float32)#.to_sparse_csr()

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp
)

with torch.no_grad():
    graphs = dynamo_compiler.importer(foo, in1, in2, in3)
    # print(graphs.print_tabular()) 


assert len(graphs) == 1
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
pattern_list = [simply_fuse]
graph.fuse_ops(pattern_list)
driver = GraphDriver(graph)
driver.subgraphs[0].lower_to_top_level_ir()
path_prefix = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(path_prefix, "subgraph0.mlir"), "w") as module_file:
    print(driver.subgraphs[0]._imported_module, file=module_file)
with open(os.path.join(path_prefix, "forward.mlir"), "w") as module_file:
    print(driver.construct_main_graph(True), file=module_file)
