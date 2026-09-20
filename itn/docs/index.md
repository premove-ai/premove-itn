---
title: Premove ITN documentation
description: Install, integrate, and deploy context-aware inverse text normalization for voice agents.
---

Premove ITN is an open-source, context-aware inverse text normalization system
for English voice-agent transcripts. It converts spoken ASR text into written
values such as phone numbers, dates, times, money, email addresses, URLs, and
identifiers.

The v0.3.0 runtime adds one-inference structured results and deterministic
temporal enrichment. Callers can request exact spans and resolved date values
without triggering a second model pass.

```text
call me at four thirty
→ call me at 04:30

the room code is one oh five
→ the room code is 105
```

## Start here

[Install Premove ITN and normalize your first transcript](/itn/docs/getting-started).

## Integrate

- [Python API](/itn/docs/python-api)
- [Command-line interface](/itn/docs/cli)
- [Supported forms](/itn/docs/supported-forms)

## Understand

- [How Premove ITN works](/itn/docs/how-it-works)
- [What is inverse text normalization?](/itn/docs/learn/what-is-inverse-text-normalization)
- [ITN for voice agents](/itn/docs/learn/itn-for-voice-agents)

## Run in production

- [Deployment](/itn/docs/deployment)
- [Troubleshooting](/itn/docs/troubleshooting)

## Go deeper

The [implementation architecture](/itn/docs/internals/architecture),
[model provenance](/itn/docs/internals/model-provenance), and other internal documents
explain how the release is built and verified.

For product context, see the [Premove ITN overview](/itn/). For measured results,
see the [benchmark](/itn/benchmarks).
