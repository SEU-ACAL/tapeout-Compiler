# import os
# from pathlib import Path
# import torch
# import torch._inductor.lowering
# import numpy as np

# from torch._inductor.decomposition import decompositions as inductor_decomp
# from buddy.compiler.frontend import DynamoCompiler
# from buddy.compiler.ops import tosa
# from buddy.compiler.graph.transform import simply_fuse
# from buddy.compiler.graph import GraphDriver

# from typing import List
# import torch._dynamo as dynamo


# # def sparse_matmul(x, y):
# #     result = torch.mm(x, y)
# #     result = torch.mm(y, result)
# #     return result

# # def foo(x, y, z):
# #     return sparse_matmul(x, y)
# #     # return torch.matmul(z, y)
# #     # return torch.matmul(x, y)

# # torch.library.define("mylib::sparse_matmul", "(Tensor x, Tensor y) -> Tensor")
# # def sparse_matmul_impl(x, y):
# #     result = torch.mm(x, y)
# #     return result # torch.mm(y, result)
# # torch.library.impl("mylib::sparse_matmul", "default", sparse_matmul_impl)




# # in1 = torch.randn(3, 4, dtype=torch.float32)
# # in2 = torch.randn(4, 3, dtype=torch.float32)

# # in3 = torch.tensor([[0, 0, 1, 0], 
# #                     [1, 2, 0, 0], 
# #                     [0, 0, 0, 0]], dtype=torch.float32).to_sparse_csr()


# def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
#     print("my_compiler() called with FX graph:")
#     gm.graph.print_tabular()
#     return gm.forward  # return a python callable

# @dynamo.optimize(my_compiler)
# # def foo(x, y, z):
# #     return torch.mm(z, y)
# def csr_spmm(row_ptr, col_idx, values, B, M, N, K):
#     # 使用torch.full代替torch.zeros或empty().fill_()
#     C = torch.full((M, N), 0, dtype=torch.float32)

#     # 将values转换为与B相同的数据类型
#     values = values.to(B.dtype)
    
#     # 获取B对应列的值
#     B_indexed = B[col_idx]  # shape: [nnz, N]
    
#     # 计算元素级乘法
#     values_expanded = values.unsqueeze(1)  # [nnz, 1]
#     weighted = torch.mul(values_expanded, B_indexed)  # [nnz, N]
    
#     # 使用循环累加到C中（避免scatter_add_）
#     for row in range(M):
#         start = row_ptr[row]
#         end = row_ptr[row+1]
        
#         # if end > start:  # 有非零元素
#         row_contrib = torch.sum(weighted[start:end], dim=0)
#         # 累加到C的对应行
#         C[row] = row_contrib
#         # else:
#         #     C[row] = 0
    
#     return C
    
#     # for row in range(M):
#     #     start = row_ptr[row]
#     #     end = row_ptr[row+1]
#     #     nnz = end - start

#     #     # if nnz == 0: 
#     #     #     continue
#     #     cols = col_idx[start:end]
#     #     vals = values[start:end] # (nnz,)

#     #     B_rows = B[cols] # (nnz, N)

#     #     B_psum = torch.mul(vals.unsqueeze(1), B_rows) # (nnz, N)

#     #     row_sum = torch.sum(B_psum, dim=0) # (N,)
#     #     C[row] = row_sum

#     # return C

# def generate_csr_int(M, K, density=0.3, max_val=1000):
#     """生成整型CSR格式稀疏矩阵"""
#     torch.manual_seed(42)
    
#     # 生成行指针
#     row_ptr = [0]
#     for _ in range(M):
#         nnz = torch.randint(0, int(K*density)+1, (1,)).item()
#         row_ptr.append(row_ptr[-1] + nnz)
    
#     total_nnz = row_ptr[-1]
#     col_idx = torch.cat([torch.randperm(K)[:row_ptr[i+1]-row_ptr[i]] 
#                        for i in range(M)])
    
#     values = torch.randint(-max_val, max_val, (total_nnz,), dtype=torch.int32)
    
#     return (torch.tensor(row_ptr, dtype=torch.int32),
#             col_idx.to(torch.int32), values)

# # 准备测试数据
# M, K, N = 16, 16, 8
# max_val = 10
# row_ptr, col_idx, values = generate_csr_int(M, K, max_val=max_val)
# B = torch.randint(-max_val, max_val, (K, N), dtype=torch.int32)

# # Initialize the dynamo compiler.
# dynamo_compiler = DynamoCompiler(
#     primary_registry=tosa.ops_registry,
#     aot_autograd_decomposition=inductor_decomp,
#     verbose=True
# )

# # print(row_ptr, col_idx, values, B, M, N, K)
# # csr_spmm(row_ptr, col_idx, values, B, M, N, K)
# # foo(in3, in2, in3)

# with torch.no_grad():
#     graphs = dynamo_compiler.importer(csr_spmm, row_ptr, col_idx, values, B, M, N, K)
#     # graphs = dynamo_compiler.importer(foo, values, B)
#     # print(graphs.print_tabular()) 


# assert len(graphs) == 1
# graph = graphs[0]
# params = dynamo_compiler.imported_params[graph]
# pattern_list = [simply_fuse]
# graph.fuse_ops(pattern_list)
# driver = GraphDriver(graph)
# driver.subgraphs[0].lower_to_top_level_ir()
# path_prefix = os.path.dirname(os.path.abspath(__file__))
# with open(os.path.join(path_prefix, "subgraph0.mlir"), "w") as module_file:
#     print(driver.subgraphs[0]._imported_module, file=module_file)
# with open(os.path.join(path_prefix, "forward.mlir"), "w") as module_file:
#     print(driver.construct_main_graph(True), file=module_file)
