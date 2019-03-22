"""
Performs an optimized version of matrix multiplication with VTA using blocking. For TVM installation instructions, visit https://docs.tvm.ai/vta/install.html.
"""

######################################################################
# RPC Setup
# ------------------------------------------------------------------

from __future__ import absolute_import, print_function

import os
import tvm
import vta
import numpy as np
from tvm import rpc
from tvm.contrib import util
from vta.testing import simulator

env = vta.get_env()

host = os.environ.get("VTA_PYNQ_RPC_HOST", "192.168.2.99")
port = int(os.environ.get("VTA_PYNQ_RPC_PORT", "9091"))

if env.TARGET == "pynq":

    assert tvm.module.enabled("rpc")
    remote = rpc.connect(host, port)

    vta.reconfig_runtime(remote)

    vta.program_fpga(remote, bitstream=None)

elif env.TARGET == "sim":
    remote = rpc.LocalSession()

######################################################################
# Computation Declaration
# ------------------------------------------------------------------

batch_size = 1
in_channels = 1024
out_channels = 1024
assert batch_size % env.BATCH == 0
assert in_channels % env.BLOCK_IN == 0
assert out_channels % env.BLOCK_OUT == 0

# Derive the tiled input tensor shapes
data_shape = (batch_size // env.BATCH,
              in_channels // env.BLOCK_IN,
              env.BATCH,
              env.BLOCK_IN)
weight_shape = (out_channels // env.BLOCK_OUT,
                in_channels // env.BLOCK_IN,
                env.BLOCK_OUT,
                env.BLOCK_IN)
output_shape = (batch_size // env.BATCH,
                out_channels // env.BLOCK_OUT,
                env.BATCH,
                env.BLOCK_OUT)
num_ops = in_channels * out_channels * batch_size * 2

# Reduction axes
ic = tvm.reduce_axis((0, in_channels // env.BLOCK_IN), name='ic')
ic_tns = tvm.reduce_axis((0, env.BLOCK_IN), name='ic_tns')

# Input placeholder tensors
data = tvm.placeholder(data_shape, name="data", dtype=env.inp_dtype)
weight = tvm.placeholder(weight_shape, name="weight", dtype=env.wgt_dtype)

# Copy buffers
data_buf = tvm.compute(data_shape,
                       lambda *i: data(*i),
                       "data_buf")
weight_buf = tvm.compute(weight_shape,
                         lambda *i: weight(*i),
                         "weight_buf")

# Declare matrix multiply computation
res_gemm = tvm.compute(output_shape,
                       lambda bo, co, bi, ci: tvm.sum(
                            data_buf[bo, ic, bi, ic_tns].astype(env.acc_dtype) *
                            weight_buf[co, ic, ci, ic_tns].astype(env.acc_dtype),
                            axis=[ic, ic_tns]),
                       name="res_gem")

# Add shift stage for fix-point normalization
res_shr = tvm.compute(output_shape,
                      lambda *i: res_gemm(*i) >> env.INP_WIDTH,
                      name="res_shr")

# Apply clipping between (0, input max value)
inp_max = (1<<(env.INP_WIDTH-1))-1
res_max = tvm.compute(output_shape,
                      lambda *i: tvm.max(res_shr(*i), 0),
                      "res_max")
res_min = tvm.compute(output_shape,
                      lambda *i: tvm.min(res_max(*i), inp_max),
                      "res_min")

# Apply typecast to input data type before sending results back
res = tvm.compute(output_shape,
                  lambda *i: res_min(*i).astype(env.inp_dtype),
                  name="res")

######################################################################
# Scheduling the Computation
# --------------------------------------------------------------------

# Create and display default TVM schedule
s = tvm.create_schedule(res.op)
print(tvm.lower(s, [data, weight, res], simple_mode=True))

######################################################################
# Blocking the Computation
# ------------------------------------------------------------------- 

# Define tiling size in terms of VTA tensor shape sizes 
b_block = 1 // env.BATCH
i_block = 256 // env.BLOCK_IN
o_block = 256 // env.BLOCK_OUT

# Tile the output tensor along the batch and output channel dimensions
# (since by default we are doing single batch inference, the split along
#  the batch dimension has no effect)
b, oc, b_tns, oc_tns = s[res].op.axis
b_out, b_inn = s[res].split(b, b_block)
oc_out, oc_inn = s[res].split(oc, o_block)
s[res].reorder(b_out, oc_out, b_inn, oc_inn)

# Move intermediate computation into each output compute tile
s[res_gemm].compute_at(s[res], oc_out)
s[res_shr].compute_at(s[res], oc_out)
s[res_max].compute_at(s[res], oc_out)
s[res_min].compute_at(s[res], oc_out)

# Apply additional loop split along reduction axis (input channel)
b_inn, oc_inn, b_tns, oc_tns = s[res_gemm].op.axis
ic_out, ic_inn = s[res_gemm].split(ic, i_block)

# Reorder axes. We move the ic_out axis all the way out of the GEMM
# loop to block along the reduction axis
s[res_gemm].reorder(ic_out, b_inn, oc_inn, ic_inn, b_tns, oc_tns, ic_tns)

# Display TVM schedule after blocking
print(tvm.lower(s, [data, weight, res], simple_mode=True))

######################################################################
# Lowering Copies to DMA Transfers
# -------------------------------------------------------------------
 
# Set scope of SRAM buffers
s[data_buf].set_scope(env.inp_scope)
s[weight_buf].set_scope(env.wgt_scope)
s[res_gemm].set_scope(env.acc_scope)
s[res_shr].set_scope(env.acc_scope)
s[res_min].set_scope(env.acc_scope)
s[res_max].set_scope(env.acc_scope)

# Block data and weight cache reads
s[data_buf].compute_at(s[res_gemm], ic_out)
s[weight_buf].compute_at(s[res_gemm], ic_out)

# Use DMA copy pragma on DRAM->SRAM operations
s[data_buf].pragma(s[data_buf].op.axis[0], env.dma_copy)
s[weight_buf].pragma(s[weight_buf].op.axis[0], env.dma_copy)

# Use DMA copy pragma on SRAM->DRAM operation
# (this implies that these copies should be performed along b_inn,
# or result axis 2)
s[res].pragma(s[res].op.axis[2], env.dma_copy)

######################################################################
# Lowering Computation to VTA Compute Intrinsics
# ------------------------------------------------------------------- 

# Apply tensorization over the batch tensor tile axis
s[res_gemm].tensorize(b_tns, env.gemm)

# Add an ALU pragma over the shift and clipping operations
s[res_shr].pragma(s[res_shr].op.axis[0], env.alu)
s[res_min].pragma(s[res_min].op.axis[0], env.alu)
s[res_max].pragma(s[res_max].op.axis[0], env.alu)

# Display final lowered TVM schedule
print(vta.lower(s, [data, weight, res], simple_mode=True))

######################################################################
# TVM Compilation and Verification
# -------------------------------------------------------------------

# Compile the TVM module
my_gemm = vta.build(s, [data, weight, res], "ext_dev", env.target_host, name="my_gemm")
temp = util.tempdir()
my_gemm.save(temp.relpath("gemm.o"))
remote.upload(temp.relpath("gemm.o"))
f = remote.load_module("gemm.o")

# Get the remote device context
ctx = remote.ext_dev(0)

# Initialize the data and weight arrays randomly in the int range of (-128, 128]
data_np = np.random.randint(
    -128, 128, size=(batch_size, in_channels)).astype(data.dtype)
weight_np = np.random.randint(
    -128, 128, size=(out_channels, in_channels)).astype(weight.dtype)

# Apply packing to the data and weight arrays from a 2D to a 4D packed layout
data_packed = data_np.reshape(batch_size // env.BATCH,
                              env.BATCH,
                              in_channels // env.BLOCK_IN,
                              env.BLOCK_IN).transpose((0, 2, 1, 3))
weight_packed = weight_np.reshape(out_channels // env.BLOCK_OUT,
                                  env.BLOCK_OUT,
                                  in_channels // env.BLOCK_IN,
                                  env.BLOCK_IN).transpose((0, 2, 1, 3))

# Format the input/output arrays with tvm.nd.array to the DLPack standard
data_nd = tvm.nd.array(data_packed, ctx)
weight_nd = tvm.nd.array(weight_packed, ctx)
res_nd = tvm.nd.array(np.zeros(output_shape).astype(res.dtype), ctx)

# Invoke the module to perform the computation
f(data_nd, weight_nd, res_nd)

# Verify against numpy implementation
res_ref = np.dot(data_np.astype(env.acc_dtype),
                 weight_np.T.astype(env.acc_dtype))
res_ref = res_ref >> env.INP_WIDTH
res_ref = np.clip(res_ref, 0, inp_max)
res_ref = res_ref.astype(res.dtype)
res_ref = res_ref.reshape(batch_size // env.BATCH,
                          env.BATCH,
                          out_channels // env.BLOCK_OUT,
                          env.BLOCK_OUT).transpose((0, 2, 1, 3))
np.testing.assert_equal(res_ref, res_nd.asnumpy())
print("Successful blocked matrix multiply test!")
