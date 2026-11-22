"""The teacher-only reminder texts (B-11), shared by the request adapter and the driver's checks.

    REMINDER=0     none
    REMINDER=1     the tagged line (pilots 2–4)
    REMINDER=rule  the same demand written as one more rule of the system prompt: no tag, never calls itself a
                   reminder (2026-11-20: the teacher wrote "The reminder asks me to …" in its thoughts)
    REMINDER=<other text>   that text
None of these may appear in a trajectory log or a training prompt: they exist only in the request sent to the teacher.
"""
import hashlib

PRESETS = {
    "1": ("<reminder>Before EVERY tool call, including simple ones like reading a file or running a command, first reason briefly "
          "in your thinking about why this step is needed and what you expect to see.</reminder>"),
    "rule": ("- **Reason Before EVERY Tool Call**: Before each tool call — including simple reads, searches and test runs — first think "
             "in 1-3 short sentences: why this step is needed and what you expect to see. NEVER emit a tool call without thinking first."),
}


def text(value):
    """REMINDER value → the text ('' = none)"""
    v = "" if value is None else str(value)
    return "" if v in ("", "0") else PRESETS.get(v, v)


def ident(value):
    """a short name for the row / the log: off · the preset's name · sha of a free text"""
    v = "" if value is None else str(value)
    return "off" if v in ("", "0") else v if v in PRESETS else "text-" + hashlib.sha256(v.encode()).hexdigest()[:8]


def needles():
    """distinctive pieces of every preset: one of them in a trajectory log line = the reminder leaked into the data"""
    return ["<reminder>", "Reason Before EVERY Tool Call", "NEVER emit a tool call without thinking first"]


# --- B-11r: text taken OUT of the teacher's system prompt (CEIL_SYS_CUT / SYS_CUT), in the outgoing request only
HIDDEN_CLAUSE = " and graded by hidden tests written from the issue specification"      # V21 prompt, first sentence
CUTS = {"hidden": HIDDEN_CLAUSE}


def cut_text(value):
    """SYS_CUT value → the text removed ('' = nothing)"""
    v = "" if value is None else str(value)
    return "" if v in ("", "0") else CUTS.get(v, v)


def cut_ident(value):
    v = "" if value is None else str(value)
    return "off" if v in ("", "0") else v if v in CUTS else "text-" + hashlib.sha256(v.encode()).hexdigest()[:8]
