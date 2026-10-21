mode: after
anchor: If edit_file reports no match or multiple matches
---
- If edit_file reports that a parameter is missing (for example `old_string`), the call itself was malformed: do not call edit_file again for that change. Make it at once with one run_command running a quoted-heredoc Python script (`python3 - <<'EOF'`): read the file, assert the old text occurs exactly once, replace it, write the file. Then continue with the next step.
