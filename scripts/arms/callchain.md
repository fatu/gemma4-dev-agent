mode: after
anchor: - If the same behaviour has parallel implementations (sync and async variants
---
- If you add or change a parameter, option, attribute or return value, follow it vertically before submitting: `git grep -n` its name and the function you changed, then update every caller that must pass it on and every callee or table that must accept or list it (defaults, `__all__`, option lists, settings, reports, docs_src examples). A change that stops at the first function usually fails the hidden tests.
