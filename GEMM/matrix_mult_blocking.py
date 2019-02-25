import tvm
import numpy
import timeit

# The size of the matrix
# (M, K) x (K, N)
M = 1024
K = 1024
N = 1024

# The default tensor type in tvm
dtype = "float32"

target = 'llvm'
ctx = tvm.context(target, 0)

# Random generated tensor for testing
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

################################################################################################
# Blocking
# --------
# A important trick to enhance the cache hit rate is blocking --- data chunk will be computed
# block by block. The memory access inside the block is a small neighbourhood which is with high
# memory locality. In this tutorial, I picked up 32 as the blocking factor. So the block will
# fill 32 * 32 * sizeof(float) which is 4KB in the cache whose total size is 32KB (L1 data cache)

bn = 32
s = tvm.create_schedule(C.op)

# Blocking by loop tiling
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
k, = s[C].op.reduce_axis
ko, ki = s[C].split(k, factor=4)

# Hoist reduction domain outside the blocking loop
s[C].reorder(xo, yo, ko, ki, xi, yi)

func = tvm.build(s, [A, B, C], target=target, name='mmult')
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype = dtype), ctx)
func(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)

# By simply tiling the loop 32x32, and hoisting ko, ki outside the blocking loops,
# we can see big speedup compared with the baseline.
evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
print('Opt1: %f' % evaluator(a, b, c).mean)

################################################################################################
# Here is the generated IR after blocking.

print(tvm.lower(s, [A, B, C], simple_mode=True))
