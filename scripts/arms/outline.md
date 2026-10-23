mode: after
anchor: - Once the target is located, read the relevant implementation plus one relevant caller or test case.
---
- For a large target file (more than about 300 lines), first list its classes and functions with line ranges in one run_command: `python3 -c "import ast,sys; t=ast.parse(open(sys.argv[1]).read()); [print(type(n).__name__, n.name, n.lineno, n.end_lineno) for n in ast.walk(t) if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]" path/to/file.py`. It lists async functions too, which the graph tools miss. Then read only the ranges you need, including every parallel sync/async variant.
