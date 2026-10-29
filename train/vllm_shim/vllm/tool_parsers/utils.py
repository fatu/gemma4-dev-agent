# the two helpers the gemma4 tool parser / abstract base import (vllm's utils.py needs partial_json_parser)
def find_common_prefix(s1: str, s2: str) -> str:
    n = 0
    for a, b in zip(s1, s2):
        if a != b:
            break
        n += 1
    return s1[:n]


def get_json_schema_from_tools(tool_choice, tools):
    return None
