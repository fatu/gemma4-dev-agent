mode: replace
anchor: ## 4. Verify (at most 2 calls)
---
## 4. Verify (at most 4 calls)
- Passing existing tests does not prove the fix: the hidden tests are new and check the exact behaviour the issue asks for. So the first check is one `python -c` run of the issue's own example (its exact inputs, names and calls) that prints the result. Compare that output with what the issue says should happen. If it differs, fix the source and run the same check again. Then run one existing test file as below.
