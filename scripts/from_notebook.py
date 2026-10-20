"""Write the submission files a Kaggle notebook embeds as `FILES = json.loads('...')` into a directory.

    python3 scripts/from_notebook.py <notebook.ipynb> <out_dir>

Several public notebooks (e.g. the LB-0.24 baseline) ship their whole submission as one JSON dict of path → text in a
code cell and write the zip from it. This recovers those files locally so eval_dev.sh can use them as SUBMISSION.
Nothing is downloaded and nothing is executed: the dict literal is parsed, not run.
"""
import ast
import json
import pathlib
import re
import sys


def embedded_files(nb_path):
    nb = json.load(open(nb_path))
    for cell in nb["cells"]:
        src = "".join(cell.get("source", ""))
        m = re.search(r"FILES = json\.loads\('(.*)'\)\n", src)
        if cell.get("cell_type") == "code" and m:
            return json.loads(ast.literal_eval("'" + m.group(1) + "'"))
    sys.exit(f"no `FILES = json.loads('...')` cell in {nb_path}")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    files, out = embedded_files(sys.argv[1]), pathlib.Path(sys.argv[2])
    for rel, text in files.items():
        p = pathlib.PurePosixPath(rel)
        if p.is_absolute() or ".." in p.parts:
            sys.exit(f"refusing path {rel}")
        dest = out / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
        print(f"{dest}  {len(text)} chars")
    if not (out / "agent.yaml").exists():
        sys.exit("no agent.yaml among the files: not a submission")


if __name__ == "__main__":
    main()
