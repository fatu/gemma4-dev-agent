mode: after
anchor: - If edit_file reports no match or multiple matches, read the exact source region and retry
---
- Guarded fallback script, the shape to use: `python3 -c "from pathlib import Path; p = Path('pkg/mod.py'); src = p.read_text(encoding='utf-8'); old = '...'; new = '...'; assert src.count(old) == 1, src.count(old); updated = src.replace(old, new, 1); compile(updated, str(p), 'exec'); p.write_text(updated, encoding='utf-8'); print('OK')"` — the assert stops a missing or ambiguous match, compile() stops a syntax error before the file is written. Then run `git diff pkg/mod.py` to confirm only the intended lines changed. Never rewrite a whole file with write_file, and never touch pytest.ini or conftest.py to hide an error.
