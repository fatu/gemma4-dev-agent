mode: after
anchor: - Once the target is located, read the relevant implementation plus one relevant caller or test case.
---
- Find a symbol where it is defined, not where it is called: `git grep -n -E 'def NAME|class NAME' -- '*.py'`, then read about 60 lines around that line with read_file start_line and end_line. Edit the definition when the wrong behaviour is produced there. Never read a whole file.
