mode: replace
anchor: Scratch files go in /tmp only. Any file left under /workspace becomes part of the graded patch.
---
Repro scripts always go to /tmp/repro.py, never under /workspace. Write it through run_command (`cat > /tmp/repro.py <<'EOF'` … `EOF`, then `python /tmp/repro.py`): write_file and edit_file reject paths outside /workspace. Any file left under /workspace becomes part of the graded patch.
