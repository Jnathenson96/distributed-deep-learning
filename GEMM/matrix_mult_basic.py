from __future__ import absolute_import, print_function

import tvm
import numpy
import timeit 


# Matrix dimensions
# (M, K) x (K, N)
M = 1024
K = 1024
N = 1024

# default tensor type in tvm
dtype = "float32"

target = 'llvm'
ctx = tvm.context(target, 0)


# generate random tensors for testing
a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), ctx)
b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), ctx)

np_repeat = 100
np_runing_time = timeit.timeit(setup='import numpy\n'
                                     'M = ' + str(M) + '\n'
                                     'K = ' + str(K) + '\n'
                                     'N = ' + str(N) + '\n'
                                     'dtype = "float32"\n'
                                     'a = numpy.random.rand(M, K).astype(dtype)\n'
                                     'b = numpy.random.rand(K, N).astype(dtype)\n',
                               stmt='answer = numpy.dot(a, b)',
                               number=np_repeat)
print("Numpy running time: %f" % (np_runing_time / np_repeat))

answer = numpy.dot(a.asnumpy(), b.asnumpy())

# Algorithm
k = tvm.reduce_axis((0, K), 'k')
A = tvm.placeholder((M, K), name='A')
B = tvm.placeholder((K, N), name='B')
C = tvm.compute(
           (M, N),
           lambda x, y: tvm.sum(A[x, k] * B[k, y], axis=k),
           name='C')

# Default schedule
s = tvm.create_schedule(C.op)
func = tvm.build(s, [A, B, C], target=target, name='mmult')
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), ctx)
func(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
print('Baseline: %f' % evaluator(a, b, c).mean)
print(tvm.lower(s, [A, B, C], simple_mode=True))

