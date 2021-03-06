import tvm
import numpy as np
import os
import timeit

#tgt="aocl_sw_emu"
tgt="aocl -device=p385a_sch_ax115"

dtype = "float32"
M = 1024

func = tvm.module.load("vmmult.so")
func_dev = tvm.module.load("vmmult.aocx")
func.import_module(func_dev)

ctx = tvm.context(tgt, 0)

a = tvm.nd.array(np.random.rand(M).astype(dtype), ctx)
b = tvm.nd.array(np.random.rand(M,M).astype(dtype), ctx)
c = tvm.nd.array(np.zeros((M,), dtype = dtype), ctx)

np_repeat = 100
np_runing_time = timeit.timeit(setup='import numpy\n'
                                     'M = ' + str(M) + '\n'
                                     'dtype = "float32"\n'
                                     'a = numpy.random.rand(M,).astype(dtype)\n'
                                     'b = numpy.random.rand(M, M).astype(dtype)\n',
                               stmt='answer = numpy.dot(a, b)',
                               number=np_repeat)

answer = np.dot(b.asnumpy(), a.asnumpy())

func(a, b, c)

tvm.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
print('Baseline: %f' % evaluator(a, b, c).mean)
print("Numpy running time: %f" % (np_runing_time / np_repeat))
