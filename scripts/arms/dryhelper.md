mode: after
anchor: - Fix the cause, not the symptom.
---
- Do not copy-paste the same fix into several entry points: put it in one shared internal helper (reuse an existing one, or extract one such as _normalize_key) and make every twin entry point call it, so all of them behave the same.
