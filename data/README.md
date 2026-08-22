# Data

`generated/` contains reproducible synthetic data and is ignored by Git.

Future reviewed data should use separate directories for development and the
locked benchmark. Never use benchmark templates in synthetic training data.
Never commit private or identifying transcript data.

Each JSONL record follows this shape:

```json
{
  "text": "meet me at four thirty",
  "tokens": ["meet", "me", "at", "four", "thirty"],
  "labels": ["O", "O", "O", "B-TIME", "I-TIME"],
  "spans": [{"start": 11, "end": 22, "kind": "TIME", "value": "04:30"}]
}
```

`start` is inclusive and `end` is exclusive. Both refer to the original text.
`value` is evaluation metadata. It is not a target for the contextual model.
