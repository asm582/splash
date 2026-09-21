# Apple9 Q4 decode with simdgroup matrices

The default Apple9 one-lane decode plan uses eight-row bfloat matrix operands.
Apple10, wider decode batches and prefill retain their existing policies.
The packed Q4 weights and the `Q4Params` ABI are unchanged.

Each SIMD group computes two 8-column fragments, or one fragment each for
gate and up. Nibbles become exact bfloat values `128 + q`; fp32 accumulation
subtracts `128 * sum(x)` before applying the existing scale and bias. The
activation table stays bfloat, including FFN down inputs that can exceed half's
range. The residual and gate/up paths retain the original bfloat rounding
boundary before their epilogues. Arithmetic is reassociated and is not bitwise
identical to the sequential Q4 kernels.

`LinearPlan` owns the grid, split count and scratch sizes. The split rule picks
powers of two up to eight to approach 16 column/K threadgroups per GPU core,
while retaining at least 12 quantization groups per partition. The final
arriving group reduces partials in a fixed order and resets its counter; no
group spins. Every matrix accumulator chain is initialized before the loop.
Initializing fragments only within the loop produced wrong results under
Apple10 GPU validation; the dedicated regression test covers this case.

The decode arena owns one reusable activation table, row-sum buffer, partial
buffer and counter buffer, charged to the memory budget. It is private to the
serial decode command stream. Counters are zeroed at allocation and reset by
each completed dispatch. Startup choices reserve the maximum of the default
and selected plans. No scratch allocation occurs while encoding a projection.

RMSNorm writes its ordinary output and the matrix operand layout from the same
rounded bfloat values. Its next Q4 consumer marks the input prepared. Other
producers use a separate preparation dispatch. Reused draft head/selector and
context inputs share their prepared table until another producer overwrites it.
This keeps the integration limited to one producer and the Q4 operator.

## Validation

```sh
make -j8 all test-engine-cpu test-engine-metal
```

The Metal target enables GPU shader validation. `q4-sgmatrix` compares against
an independent fp64 packed-weight reference for all three epilogues, every
valid split count in 1/2/4/8, small and irregular K, and real FFN widths. Inputs
include values above half's range, cancellation and zero weights. It also
checks guard bytes, repeated workspace use, deterministic reductions, zeroed
counters, and bitwise equality of separate versus fused RMSNorm preparation.

Error bounds use input and weight magnitudes, so cancellation does not hide
behind a relative-output threshold. They cover fp32 dot/affine reassociation,
bfloat output rounding and epilogue propagation. The tuning qualification uses
the same operand-based reasoning; its timing includes preparation when the
input has no fused producer. End-to-end measurements are required to assess
the benefit of producer fusion and changes in speculative acceptance.
