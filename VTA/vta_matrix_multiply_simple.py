"""
Basic matrix multiplication with VTA. Will not work unless VTA is installed first. For installation instructions, visit https://docs.tvm.ai/vta/install.html. 
"""

from __future__ import absolute_import, print_function

import os
import tvm
import vta
import numpy as np
from tvm import rpc
from tvm.contrib import util
from vta.testing import simulator

##################################################################
# RPC setup
# ---------------------------------------------------------------
 
# Load VTA parameters from vta/config/vta_config.json 
env = vta.get_env()

# Get Pynq RPC host IP address and port number from the OS environment
host = os.environ.get("VTA_PYNQ_RPC_HOST", "192.168.2.99")
port = int(os.environ.get("VTA_PYNQ_RPC_PORT", "9091"))

if env.TARGET == "pynq":

    # Make sure that TVM was compiled with RPC=1
    assert tvm.module.enabled("rpc")
    remote = rpc.connect(host, port)

    # Reconfigure the JIT runtime
    vta.reconfig_runtime(remote)

    # Program the FPGA with a pre-compiled VTA bitstream. If you wish to program the FPGA with your own custom bitstream, pass the path to the bistream file instead of 'None'.
    vta.program_fpga(remote, bitstream=None)

# In simulation mode, host the RPC server locally.
elif env.TARGET == "sim":
    remote = rpc.LocalSession()

######################################################################
# Data dimensions 
# -------------------------------------------------------------------

# m, n, and o represent the shape of the matrix multiplication 

# Output channel factor m - total 16x16=256 output channels
m = 16
# Input channel factor n - total 16x16=256 input channels
n = 16
# Batch factor o (we use single batch inference)
o = 1

# A and B are the input tensors

# A placeholder tensor in tiled data format
A = tvm.placeholder((o, n, env.BATCH, env.BLOCK_IN), name="A", dtype=env.inp_dtype)
# B placeholder tensor in tiled data format
B = tvm.placeholder((m, n, env.BLOCK_OUT, env.BLOCK_IN), name="B", dtype=env.wgt_dtype)
# A copy buffer
A_buf = tvm.compute((o, n, env.BATCH, env.BLOCK_IN), lambda *i: A(*i), "A_buf")
# B copy buffer
B_buf = tvm.compute((m, n, env.BLOCK_OUT, env.BLOCK_IN), lambda *i: B(*i), "B_buf")

######################################################################
# Matrix Multiplication
# -------------------------------------------------------------------

# Outer input feature reduction axis
ko = tvm.reduce_axis((0, n), name="ko")
# Inner input feature reduction axis
ki = tvm.reduce_axis((0, env.BLOCK_IN), name="ki")
# Describe the in-VTA matrix multiplication
C_buf = tvm.compute(
    (o, m, env.BATCH, env.BLOCK_OUT),
    lambda bo, co, bi, ci:
        tvm.sum(A_buf[bo, ko, bi, ki].astype(env.acc_dtype) *
                B_buf[co, ko, ci, ki].astype(env.acc_dtype),
                axis=[ko, ki]),
    name="C_buf")

######################################################################
# Cast results of VTA computation to main memory
# ------------------------------------------------------------------
 
C = tvm.compute(
    (o, m, env.BATCH, env.BLOCK_OUT),
    lambda *i: C_buf(*i).astype(env.inp_dtype),
    name="C")

######################################################################
# Schedule computation 
# ------------------------------------------------------------------ 

# Declare and print the default (non-optimized) schedule 
# This schedule makes sense but will not yet compile on VTA
s = tvm.create_schedule(C.op)
print(tvm.lower(s, [A, B, C], simple_mode=True))

# Set the intermediate tensor's scope to VTA's on-chip buffers
s[A_buf].set_scope(env.inp_scope)
s[B_buf].set_scope(env.wgt_scope)
s[C_buf].set_scope(env.acc_scope)

# Move buffer copy into matrix multiply loop
s[A_buf].compute_at(s[C_buf], ko)
s[B_buf].compute_at(s[C_buf], ko)

# Tag the buffer copies with the DMA pragma to insert a DMA transfer
s[A_buf].pragma(s[A_buf].op.axis[0], env.dma_copy)
s[B_buf].pragma(s[B_buf].op.axis[0], env.dma_copy)
s[C].pragma(s[C].op.axis[0], env.dma_copy)

# Tensorization - analogous to vectorization, but extended to a higher-dimensional unit of computation
s[C_buf].reorder(
    ko,
    s[C_buf].op.axis[0],
    s[C_buf].op.axis[1],
    s[C_buf].op.axis[2],
    s[C_buf].op.axis[3],
    ki)
s[C_buf].tensorize(s[C_buf].op.axis[2], env.gemm)

# Finalized schedule
print(vta.lower(s, [A, B, C], simple_mode=True))

######################################################################
# TVM Compilation
# -------------------------------------------------------------------

# Build GEMM VTA kernel
my_gemm = vta.build(s, [A, B, C], "ext_dev", env.target_host, name="my_gemm")

# Write the compiled module into an object file.
temp = util.tempdir()
my_gemm.save(temp.relpath("gemm.o"))

# Send the executable over RPC
remote.upload(temp.relpath("gemm.o"))

# Load the compiled module
f = remote.load_module("gemm.o")

######################################################################
# Running the Function
# ------------------------------------------------------------------

# Get the remote device context
ctx = remote.ext_dev(0)

# Initialize the A and B arrays randomly in the int range of (-128, 128]
A_orig = np.random.randint(
    -128, 128, size=(o * env.BATCH, n * env.BLOCK_IN)).astype(A.dtype)
B_orig = np.random.randint(
    -128, 128, size=(m * env.BLOCK_OUT, n * env.BLOCK_IN)).astype(B.dtype)

# Apply packing to the A and B arrays from a 2D to a 4D packed layout
A_packed = A_orig.reshape(
    o, env.BATCH, n, env.BLOCK_IN).transpose((0, 2, 1, 3))
B_packed = B_orig.reshape(
    m, env.BLOCK_OUT, n, env.BLOCK_IN).transpose((0, 2, 1, 3))

# Format the input/output arrays with tvm.nd.array to the DLPack standard
A_nd = tvm.nd.array(A_packed, ctx)
B_nd = tvm.nd.array(B_packed, ctx)
C_nd = tvm.nd.array(np.zeros((o, m, env.BATCH, env.BLOCK_OUT)).astype(C.dtype), ctx)

# Invoke the module to perform the computation
f(A_nd, B_nd, C_nd)

######################################################################
# Verifying Correctness
# ------------------------------------------------------------------

# Compute reference result with numpy
C_ref = np.dot(A_orig.astype(env.acc_dtype),
               B_orig.T.astype(env.acc_dtype)).astype(C.dtype)
C_ref = C_ref.reshape(
    o, env.BATCH, m, env.BLOCK_OUT).transpose((0, 2, 1, 3))
np.testing.assert_equal(C_ref, C_nd.asnumpy())
print("Successful matrix multiply test!")
