import tvm

tgt_host="llvm"
#tgt="aocl_sw_emu"
tgt="aocl -device=p385a_sch_ax115"

M = 1024 #MxM for matrix, MxK for vector
K = 1024
N = 1

# Algorithm
k = tvm.reduce_axis((0, K), 'k')
A = tvm.placeholder((M, K), name='A')
B = tvm.placeholder((K, N), name='B')
C = tvm.compute(
           (M, N),
           lambda x, y: tvm.sum(A[x, k] * B[k, y], axis=k),
           name='C')

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
