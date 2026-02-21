import triton
import triton.language as tl
from jax._src import test_util as jtu


@triton.jit
def approx_tanh(x: tl.tensor):
    tl.static_assert(x.dtype == tl.float32)
    [result] = tl.inline_asm_elementwise(
        asm='tanh.approx.f32 $0, $1;',
        constraints='=f,f',
        args=[x],
        dtype=[tl.float32],
        is_pure=True,
        pack=1,
    )
    return result



if not jtu.is_cuda_compute_capability_at_least('9.0'):
    tanh = approx_tanh
else:
    tanh = tl.extra.cuda.libdevice.tanh
