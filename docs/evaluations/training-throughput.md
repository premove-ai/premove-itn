# Training throughput benchmark

The full-Google hot-path optimization was measured on Apple MPS in fp32 from
the unchanged 10k checkpoint. The benchmark used the first 100 deterministic
full-run batches: 800 examples with batch size 8. It performed optimizer
updates in memory and wrote no checkpoint.

| Implementation | Examples/second | Notes |
| --- | ---: | --- |
| Original full-run observation | approximately 2.0 | Serialized preparation and host-visible hot-path checks |
| Optimized, 100 steps | 4.856 | Two preparation workers and an eight-batch ordered queue |
| Prepared batches, legacy AdamW | 4.593 | Controlled 100-batch fused qualification baseline |
| Prepared batches, fused AdamW | 5.253 | Same checkpoint, batches, RNG seed, fp32, and update count |

The optimized run processed 800 examples in 164.736 seconds with mean
structured loss 0.129874. This is a short-run estimate, not a guaranteed
full-epoch rate. Sentence and candidate distributions can change throughput.

The optimization does not change model capacity, fp32 precision, batch size,
batch order, candidate order, structured loss, AdamW algorithm or
hyperparameters, or update count. It:

- keeps source-character path topology on the CPU;
- removes host-visible finite branches inside path dynamic programming;
- validates candidate metadata during CPU collation;
- transfers the loss accumulator only at reporting boundaries; and
- overlaps deterministic candidate/token preparation with MPS execution.

The fused-AdamW qualification on 31 August 2026 used ten warm-up batches and
100 measured batches from the unchanged 10k checkpoint. Fused execution took
152.293 seconds versus 174.173 seconds for legacy execution, a 1.144x speedup.
Both variants completed 97 optimizer steps. Their maximum relative chunk-loss
difference was `7.195e-5`, and the maximum difference across selected parameter
probes was `3.986e-7`. A fused checkpoint was saved, restored with its Adam
moments and MPS step tensors, and advanced through another optimizer step. The
raw result is stored at
`data/generated/fused_adamw_qualification/metrics.json`.

Fused AdamW changes kernel execution and floating-point operation ordering, but
does not change the AdamW algorithm or its hyperparameters. Legacy checkpoints
are migrated before optimizer loading so their saved `fused` metadata cannot
disable the qualified path.

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
