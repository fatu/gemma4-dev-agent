mode: after
anchor: ## 4. Verify (at most 2 calls)
---
- One targeted test run before `submit_patch` is required, by call 20 at the latest: run the test file closest to your change. If it fails because of your change, fix the source once and rerun once, then submit. Only if no test file is related, run one `python -c` check that exercises the issue's example instead.
