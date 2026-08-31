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
40.1 active hours. This estimate does not include checkpoint serialization and
must not be decremented using wall time across system sleep. Use the atomic live
progress artifact for the operational ETA after relaunch.

The first operational full-data launch on 30 August 2026 produced no full-run
checkpoint or result. Its 10,000-optimizer-step checkpoint interval was too
coarse, it had no live batch counter, and the MPS process stopped completing a
gradient-clipping synchronization after emergency sleep. The run was canceled
and is not an experiment result. The repaired runner checkpoints every 1,000
completed batches, persists progress every 25 batches, retains the previous
checkpoint generation, and uses an external stale-progress watchdog.
