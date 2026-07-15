# Contributing

Changes must preserve the non-linguistic core boundary, pass CI and include a
research-ledger entry when they alter a model, objective, dataset or evaluation.

1. Create a branch from the relevant model-family branch.
2. Register the hypothesis before opening target results.
3. Add or update tests.
4. Run `ruff check .`, `mypy src` and `pytest`.
5. Open a pull request with the hypothesis ID in the description.

Do not report reconstructed, cherry-picked or unregistered metrics.
