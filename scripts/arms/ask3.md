mode: after
anchor: - Use edit_file for changes to existing code.
---
- Before editing a function, answer three questions. (1) Who calls me: `git grep -n 'func_name('` for every call site; a new parameter needs a default and the signature stays compatible. (2) What do I call: read the helpers and the data structures it uses (dicts, constants, their key format) before writing conversion logic. (3) Did unmentioned inputs change: with one `python -c`, compare the return value before and after your change for None, an empty string, an int and an unknown key. Keep current behaviour for inputs the issue does not mention.
