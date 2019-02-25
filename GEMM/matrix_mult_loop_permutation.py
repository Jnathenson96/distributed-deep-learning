import tvm
import numpy
import timeit
# The size of the matrix
# (M, K) x (K, N)
# You are free to try out different shapes, sometimes TVM optimization outperforms numpy with MKL.
M = 1024
K = 1024
N = 1024

# The default tensor type in tvm
dtype = "float32"

# using Intel AVX2(Advanced Vector Extensions) ISA for SIMD
# To get the best performance, please change the following line
# to llvm -mcpu=core-avx2, or specific type of CPU you use
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

###################################################################################################
# Loop Permutation
# ----------------
# If we look at the above IR, we can see the inner loop row data is vectorized and
# B is transformed into PackedB. The traversal of PackedB is sequential now.
# So we will look at the access pattern of A. In current schedule, A is accessed column by column
# which is not cache friendly. If we change the nested loop order of ki and inner axes xi,
# the access pattern for A matrix is more cache friendly.
bn = 32
s = tvm.create_schedule(C.op)
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
k, = s[C].op.reduce_axis
ko, ki = s[C].split(k, factor=4)

# re-ordering
s[C].reorder(xo, yo, ko, xi, ki, yi)
s[C].vectorize(yi)

func = tvm.build(s, [A, B, C], target=target, name='mmult')
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype = dtype), ctx)
func(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
print('Opt3: %f' % evaluator(a, b, c).mean)
print(tvm.lower(s, [A, B, C], simple_mode=True))
