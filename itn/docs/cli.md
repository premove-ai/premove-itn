---
title: Premove ITN command-line interface
sidebarTitle: Command-line interface
description: Normalize direct text and newline-delimited transcripts with the Premove ITN CLI.
---

The CLI uses the same `PremoveITN.normalize()` implementation as the Python API.

## Normalize one transcript

```bash
premove-itn "call me at four thirty"
```

```text
call me at 04:30
```

Select a device with `--device`:

```bash
premove-itn --device cpu "the total is twenty dollars"
```

Valid values are `auto`, `cpu`, `mps`, and `cuda`.

## Process stdin

When the text argument is omitted, the CLI reads newline-delimited transcripts
from stdin:

```bash
printf 'call me at four thirty\nthe total is twenty dollars\n' | premove-itn
```

```text
call me at 04:30
the total is $20
```

Stdin mode loads the model once and processes each line in order. Use this mode
instead of starting a new process for every transcript.

Run `premove-itn --help` for command help or `premove-itn --version` for the
installed package version. Normal output uses stdout. Diagnostics use stderr.

The command requires a text argument or piped stdin. A model-load or
normalization error stops processing and returns a non-zero exit status. For
startup and device limits, see [Deployment](/itn/docs/deployment).
