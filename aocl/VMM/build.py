import tvm

tgt_host="llvm"
#tgt="aocl_sw_emu"
tgt="aocl -device=p385a_sch_ax115"

M = 1024  #MxM for matrix, MxK for vector

#A = tvm.placeholder((n,), name='A')
#B = tvm.placeholder((n,), name='B')
#C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")
m = tvm.reduce_axis((0, M), 'm')
B = tvm.placeholder((M,M), name='B') #3x3 matrix
A = tvm.placeholder((M,), name='A') #vector

C = tvm.compute(A.shape, lambda x: tvm.sum(B[x,m] * A[m], axis=m), name='C')

s = tvm.create_schedule(C.op)
px, x = s[C].split(C.op.axis[0], nparts=1)

s[C].bind(px, tvm.thread_axis("pipeline"))

func = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name="vmmult")
dev_module = func.imported_modules[0]
print("PRINTING DEV MODULE")
print(dev_module.get_source())
print("PRINTING LOW LEVEL IR")
print(tvm.lower(s,[A,B,C], simple_mode=True))

func.save("vmmult.o")
func.imported_modules[0].save("vmmult.aocx")

tvm.contrib.cc.create_shared("vmmult.so", ["vmmult.o"])
