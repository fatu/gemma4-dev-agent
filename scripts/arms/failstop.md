mode: after
anchor: - If a test related to your change fails, read the traceback, fix the source once, and rerun once.
---
- A failing check means your patch is broken: a pytest exit code other than 0, "no tests ran" or exit code 5 (your -k matched nothing), or an ImportError, IndentationError or SyntaxError from a file you edited. It is never an environment problem. Do not submit it: fix or revert your edit, then run the check again.
