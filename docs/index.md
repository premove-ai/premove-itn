---
title: Premove ITN documentation
description: Install, integrate, and deploy context-aware inverse text normalization for voice agents.
---

Premove ITN is an open-source, context-aware inverse text normalization system
for English voice-agent transcripts. It converts spoken ASR text into written
values such as phone numbers, dates, times, money, email addresses, URLs, and
identifiers.

```text
call me at four thirty
→ call me at 04:30

the room code is one oh five
→ the room code is 105
```

## Start here

[Install Premove ITN and normalize your first transcript](/docs/getting-started).

## Integrate

- [Python API](/docs/python-api)
- [Command-line interface](/docs/cli)
- [Supported forms](/docs/supported-forms)

## Understand

- [How Premove ITN works](/docs/how-it-works)
- [What is inverse text normalization?](/docs/learn/what-is-inverse-text-normalization)
- [ITN for voice agents](/docs/learn/itn-for-voice-agents)

## Run in production

- [Deployment](/docs/deployment)
- [Troubleshooting](/docs/troubleshooting)

## Go deeper

The [implementation architecture](/docs/internals/architecture),
[model provenance](/docs/internals/model-provenance), and other internal documents
explain how the release is built and verified.

For product context, see the [Premove ITN overview](/). For measured results,
see the [benchmark](/benchmarks).
