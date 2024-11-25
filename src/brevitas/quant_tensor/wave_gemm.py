import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
from iree.turbine.kernel.lang.global_symbols import *
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.utils import device_randn
from iree.turbine.kernel.wave.utils import device_zeros
from iree.turbine.kernel.wave.utils import get_default_run_config
from iree.turbine.kernel.wave.utils import get_mfma_load_elems_per_thread
from iree.turbine.kernel.wave.utils import get_mfma_store_elems_per_thread
import torch

dtype_dict = {
    'F16': {
        'MMA': MMAType.F32_32x32x8_F16, 'input_dtype': tkl.f16, 'output_dtype': tkl.f32, 'torch_output_dtype': torch.float32},
    'I8': {
        'MMA': MMAType.I32_32x32x8_I8, 'input_dtype': tkl.i8, 'output_dtype': tkl.i32, 'torch_output_dtype': torch.int32},}


def batched_gemm(a, b, kwargs):
    MMA = kwargs['MMA']
    input_dtype = kwargs['input_dtype']
    output_dtype = kwargs['output_dtype']
    torch_output_dtype = kwargs['torch_output_dtype']
    # Input sizes
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(2, 2, 1), vector_shapes={B: 0}, mma_type=MMA)]

    @tkw.wave(constraints)
    def batched_gemm(
        a: tkl.Memory[B, M, K, ADDRESS_SPACE, input_dtype],
        b: tkl.Memory[B, N, K, ADDRESS_SPACE, input_dtype],
        c: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, output_dtype],
    ):
        c_reg = tkl.Register[B, M, N, output_dtype](0.0)

        @tkw.reduction(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[B, M, N, output_dtype]) -> tkl.Register[B, M, N, output_dtype]:
            a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            #a_reg = tkw.cast(a_reg, tkl.f8e4m3fnuz)
            b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)

            #b_reg = tkw.cast(b_reg, tkl.f8e4m3fnuz)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    batch = a.shape[0]
    first_dim = a.shape[-2]
    shared_dim = a.shape[-1]
    second_dim = b.shape[-2]

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(MMA),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(MMA),
        BLOCK_B: 1,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        B: batch,
        M: first_dim,
        N: second_dim,
        K: shared_dim,
        READ_SHARED_DELAY: 1,
        WRITE_SHARED_DELAY: 1,
        READ_GLOBAL_DELAY: 2,
        WRITE_GLOBAL_DELAY: 2,
        MMA_DELAY: 1,
        VALU_DELAY: 1,
        SHUFFLE_DELAY: 1,
        SHARED_MEMORY_UNITS: 4,
        GLOBAL_MEMORY_UNITS: 4,
        MMA_UNITS: 4,
        VALU_UNITS: 8,
        SHUFFLE_UNITS: 8,}
    config = get_default_run_config()
    # if run_bench:
    #     config["benchmark_batch_size"] = 10
    #     config["benchmark_repetitions"] = 3
    # if dump_perf is not None:
    #     perf_filename = request.node.name + ".json"
    #     config["benchmark_results_file"] = os.path.join(
    #         dump_perf, "tk_" + perf_filename
    #     )
    with tk.gen.TestLaunchContext(
            hyperparams,
            canonicalize=True,
            run=True,
            run_bench=False,
            run_config=config,
            schedule=False,
            use_scheduling_barriers=False,
    ):

        c = device_zeros(batch, first_dim, second_dim).to(torch_output_dtype)
        mb = batched_gemm(a, b, c)

    return c


a = torch.randn(1, 768, 768, dtype=torch.float16).cuda()
b = torch.randn(1, 768, 768, dtype=torch.float16).cuda()
batched_gemm(a, b)
for _ in range(1):
    batched_gemm(a, b)
# print(torch.bmm(a,b.transpose(1,2)))
