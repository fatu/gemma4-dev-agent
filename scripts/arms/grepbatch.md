mode: after
anchor: - Use exact paths from the issue directly.
---
- First three calls (standard locate procedure): call 1 = one run_command `git grep -n -E 'SymbolA|SymbolB|error text' -- '*.py' | head -40 && echo '=== TEST FILES ===' && find . -maxdepth 4 -iname '*test*keyword*' | head -20` (escape regex characters such as $ . ( with a backslash) to get the source file, line numbers and the test file; call 2 = the outline of the target file, `git grep -n -E '^\s*(async )?def |^\s*class ' path/to/file.py` (or the ast outline below for a large file); call 3 = read_file with start_line and end_line on the target function, its twins and the test assertions. By the end of call 3 you should know where to edit.
