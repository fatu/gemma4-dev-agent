# vLLM parser shim (laptop only)

`train/parser_check.py --wheel harness/wheelhouse/vllm-0.19.1-*.whl` copies the *real* gemma4 reasoning and tool
parser modules out of the competition's vLLM wheel (vllm/reasoning/{abs_reasoning_parsers,basic_parsers,
gemma4_reasoning_parser,gemma4_utils}.py, vllm/tool_parsers/{abstract_tool_parser,gemma4_tool_parser,gemma4_utils}.py)
into a temp dir and lays these stub modules next to them, so the parsers import without torch / CUDA. The stubs replace
only data models and helpers (pydantic request/response classes, logger, tokenizer type alias, tool-id generator);
no parsing logic is reimplemented. On the pod `parser_check.py` imports the installed vLLM instead — that run is the
authoritative A5 number.
