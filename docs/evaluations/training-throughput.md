# Training throughput benchmark

The full-Google hot-path optimization was measured on Apple MPS in fp32 from
the unchanged 10k checkpoint. The benchmark used the first 100 deterministic
full-run batches: 800 examples with batch size 8. It performed optimizer
updates in memory and wrote no checkpoint.

| Implementation | Examples/second | Notes |
| --- | ---: | --- |
| Original full-run observation | approximately 2.0 | Serialized preparation and host-visible hot-path checks |
| Optimized, 100 steps | 4.856 | Two preparation workers and an eight-batch ordered queue |

The optimized run processed 800 examples in 164.736 seconds with mean
structured loss 0.129874. This is a short-run estimate, not a guaranteed
full-epoch rate. Sentence and candidate distributions can change throughput.

The optimization does not change model capacity, fp32 precision, batch size,
batch order, candidate order, structured loss, AdamW settings, learning rate,
or update count. It:

- keeps source-character path topology on the CPU;
- removes host-visible finite branches inside path dynamic programming;
- validates candidate metadata during CPU collation;
- transfers the loss accumulator only at reporting boundaries; and
- overlaps deterministic candidate/token preparation with MPS execution.

At the measured short-run rate, 701,135 remaining examples would take about
40.1 hours. Use progress from the first durable checkpoint for the operational
ETA after relaunch.
