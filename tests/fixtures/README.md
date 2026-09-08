# Normalization regression fixture

`normalization_regression.json` is test data for validating normalization
behavior and measuring latency changes. It protects candidate-generation output
when the normalization implementation changes.

This fixture is not training data. It is not an evaluation benchmark and must
not be used to report model quality or comparative accuracy. Training and model
evaluation code must not load it.
