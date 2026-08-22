# Architecture

## Boundary

The contextual tagger answers only two questions:

1. Which source-text span is normalizable?
2. Which deterministic parser should handle it?

It does not construct output text.

```text
TranscriptSnapshot
        |
        v
ContextualTagger ---- low confidence ----> preserve source
        |
        v
TaggedSpan[]
        |
        v
SpanRealizer (class-specific deterministic parser)
        |
        v
NormalizedSpan[]
        |
        v
TextRewriter (replace accepted spans only)
```

Streaming logic stays outside these components. The initial system recomputes
from the newest complete STT hypothesis. It does not cache Transformer state.

## Initial semantic classes

`CARDINAL`, `ORDINAL`, `DIGIT_SEQUENCE`, `DECIMAL`, `FRACTION`, `MONEY`,
`PERCENTAGE`, `DATE`, `TIME`, `DURATION`, `MEASUREMENT`, `PHONE`,
`ALPHANUMERIC`, and `ELECTRONIC`.

These are representation classes, not business meanings. `ORDER_ID` and
`BOOKING_ID` belong in contextual text, not in the label set.

## Coordinates

The pipeline has three coordinate systems:

- original-text character offsets;
- whitespace-level word tokens for the data contract;
- model subword tokens during training and inference.

Final spans always use original-text offsets. A tokenizer integration must keep
its `word_ids()` mapping and map predictions back before realization.

## Decoding and abstention

The model emits per-token logits. A constrained decoder must reject illegal BIO
transitions such as `O -> I-TIME` and `B-DATE -> I-MONEY`.

Softmax scores are not assumed to be calibrated. A held-out calibration set
will determine temperature and the acceptance threshold. The target is high
precision. Rejected spans preserve their original text.

## Runtime direction

Training can use Python, PyTorch, and Transformers. A successful model will be
exported to ONNX and quantized. A production runtime should be in-process and
combine the small model with deterministic parsers. It should not require
PyTorch, NeMo, CUDA, Docker, or a Python subprocess.
