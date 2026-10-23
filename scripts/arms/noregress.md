mode: after
anchor: - Change only what the issue requires.
---
- Zero regression: inputs the issue does not mention keep their current behaviour, and `__init__.py` / `__all__` still export every public name, including any new one.
