mode: replace
anchor: Scratch files go in /tmp only. Any file left under /workspace becomes part of the graded patch.
---
Do not create scratch files. `write_file` and `edit_file` only accept paths inside /workspace (a /tmp path is rejected and wastes a call), so run throwaway checks with `python -c` or a quoted heredoc through run_command. Any file left under /workspace becomes part of the graded patch.
