"""Apply a prompt-rule arm (scripts/arms/<name>.md) to a submission's prompts/system.md.

Rule file = a small header, `---`, then the rule text:
  mode: after | replace     after: insert the rule as a new line after the line holding `anchor`
  anchor: <exact text>      replace: swap the exact `anchor` text for the rule
  anchor_alt: <exact text>  optional: used when `anchor` is missing (the same line in an older base prompt)
If the anchor is missing (another base prompt) the rule is appended under "# EXTRA RULES", and that is printed.
Usage: python3 scripts/arms/apply.py <name> <system.md>
"""
import pathlib, sys

name, target = sys.argv[1], pathlib.Path(sys.argv[2])
head, rule = (pathlib.Path(__file__).parent / f"{name}.md").read_text().split("\n---\n", 1)
meta = dict(l.split(": ", 1) for l in head.strip().splitlines())
rule = rule.strip()
assert "{" not in rule and "}" not in rule, "ADK templates {name} from session state"
s = target.read_text()
anchor = meta["anchor"]
if anchor not in s and meta.get("anchor_alt", "\0") in s:
    anchor = meta["anchor_alt"]
    name += " (anchor_alt)"
if anchor not in s:
    s, how = s.rstrip() + "\n\n# EXTRA RULES\n" + (rule if rule.startswith("- ") else "- " + rule) + "\n", "appended (anchor not found)"
elif meta["mode"] == "replace":
    s, how = s.replace(anchor, rule, 1), "replaced the anchor"
else:
    lines = s.splitlines(keepends=True)
    i = next(i for i, l in enumerate(lines) if anchor in l)
    lines.insert(i + 1, rule + "\n")
    s, how = "".join(lines), f"inserted after line {i + 1}"
target.write_text(s)
print(f"{name}: {how}")
