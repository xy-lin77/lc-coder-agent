#!/usr/bin/env python3
"""
Sandboxed Python code executor for GRPO reward computation.
Runs untrusted model-generated code in isolated processes with resource limits.
"""

import ast
import multiprocessing
import os
import re
import signal
import sys
import textwrap
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Worker function (runs inside a fresh subprocess)
# ---------------------------------------------------------------------------

def _worker(
    code_str: str,
    fn_name: str,
    args: tuple,
    kwargs: dict,
    result_queue: multiprocessing.Queue,
) -> None:
    """
    Execute untrusted code in a resource-limited subprocess.

    Resource limits applied inside the worker so they only affect this process:
      - CPU time: 5 s hard limit
      - Virtual address space: 512 MiB

    A restricted namespace is provided that covers the standard LeetCode import
    surface (math, collections, heapq, bisect, functools, itertools, typing).
    All I/O is redirected to /dev/null so stray print statements do not pollute
    the parent's stdout/stderr.
    """
    import io
    import math
    import bisect
    import collections
    import functools
    import heapq
    import itertools
    import typing
    import resource

    # ------------------------------------------------------------------
    # Apply resource limits
    # ------------------------------------------------------------------
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    except (ValueError, resource.error):
        pass  # Not critical; timeout in parent will still fire

    try:
        mem = 512 * 1024 * 1024  # 512 MiB
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    except (ValueError, resource.error):
        pass

    # ------------------------------------------------------------------
    # Redirect stdout / stderr to /dev/null
    # ------------------------------------------------------------------
    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    sys.stderr = devnull

    # ------------------------------------------------------------------
    # Build restricted execution namespace
    # ------------------------------------------------------------------
    safe_builtins = {
        # numeric / type constructors
        "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool,
        "callable": callable, "chr": chr, "complex": complex,
        "dict": dict, "divmod": divmod, "enumerate": enumerate,
        "filter": filter, "float": float, "format": format,
        "frozenset": frozenset, "getattr": getattr, "hasattr": hasattr,
        "hash": hash, "hex": hex, "id": id, "int": int, "isinstance": isinstance,
        "issubclass": issubclass, "iter": iter, "len": len, "list": list,
        "map": map, "max": max, "min": min, "next": next, "object": object,
        "oct": oct, "ord": ord, "pow": pow, "print": print,
        "range": range, "repr": repr, "reversed": reversed,
        "round": round, "set": set, "setattr": setattr, "slice": slice,
        "sorted": sorted, "str": str, "sum": sum, "super": super,
        "tuple": tuple, "type": type, "vars": vars, "zip": zip,
        # exceptions that code may legitimately raise/catch
        "Exception": Exception, "ValueError": ValueError,
        "TypeError": TypeError, "IndexError": IndexError,
        "KeyError": KeyError, "StopIteration": StopIteration,
        "RuntimeError": RuntimeError, "OverflowError": OverflowError,
        "ZeroDivisionError": ZeroDivisionError,
        "NotImplementedError": NotImplementedError,
        "AttributeError": AttributeError,
        # needed for class definitions
        "__build_class__": __build_class__,
        "__name__": "__main__",
    }

    namespace = {
        "__builtins__": safe_builtins,
        "math": math,
        "collections": collections,
        "heapq": heapq,
        "bisect": bisect,
        "functools": functools,
        "itertools": itertools,
        "typing": typing,
        "List": typing.List,
        "Dict": typing.Dict,
        "Tuple": typing.Tuple,
        "Set": typing.Set,
        "Optional": typing.Optional,
        "Any": typing.Any,
    }

    # ------------------------------------------------------------------
    # Execute the code string
    # ------------------------------------------------------------------
    try:
        exec(compile(code_str, "<generated>", "exec"), namespace)  # noqa: S102
    except Exception as exc:
        result_queue.put(("err", f"CompilationError: {type(exc).__name__}: {exc}"))
        return

    # ------------------------------------------------------------------
    # Look up the target function
    # ------------------------------------------------------------------
    fn = namespace.get(fn_name)
    if fn is None:
        result_queue.put(("err", f"FunctionNotFound: '{fn_name}' not defined after exec"))
        return

    # ------------------------------------------------------------------
    # Call the function
    # ------------------------------------------------------------------
    try:
        result = fn(*args, **kwargs)
        result_queue.put(("ok", result))
    except Exception as exc:
        result_queue.put(("err", f"RuntimeError: {type(exc).__name__}: {exc}"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute_with_timeout(
    code: str,
    fn_name: str,
    args: tuple,
    kwargs: dict | None = None,
    timeout: int = 5,
) -> tuple[str, Any]:
    """
    Spawn a subprocess to execute *code* and call *fn_name*(*args, **kwargs).

    Returns
    -------
    ('ok', result)  – function executed without error
    ('err', message) – any failure (compile error, runtime error, timeout, OOM, …)
    """
    if kwargs is None:
        kwargs = {}

    result_queue: multiprocessing.Queue = multiprocessing.Queue()

    proc = multiprocessing.Process(
        target=_worker,
        args=(code, fn_name, args, kwargs, result_queue),
        daemon=True,
    )
    proc.start()
    proc.join(timeout)

    # ------------------------------------------------------------------
    # If still running after timeout → kill
    # ------------------------------------------------------------------
    if proc.is_alive():
        try:
            os.kill(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        proc.join(1)
        if proc.is_alive():
            try:
                os.kill(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            proc.join(0.5)
        return ("err", f"TimeoutError: execution exceeded {timeout}s")

    # ------------------------------------------------------------------
    # Drain the queue
    # ------------------------------------------------------------------
    try:
        status, payload = result_queue.get_nowait()
        return (status, payload)
    except Exception:
        exit_code = proc.exitcode
        if exit_code is not None and exit_code < 0:
            sig = -exit_code
            return ("err", f"KilledBySignal: signal {sig} (likely OOM or CPU limit)")
        return ("err", "UnknownError: worker exited without writing to queue")


def extract_function_name(code: str) -> Optional[str]:
    """
    Parse *code* with the AST and return the name of the first top-level
    ``def`` statement, or ``None`` if parsing fails or there are no functions.
    """
    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and isinstance(node.col_offset, int):
            # Only top-level functions (col_offset == 0)
            if node.col_offset == 0:
                return node.name
    return None


def extract_code_block(response: str) -> Optional[str]:
    """
    Extract a Python code block from *response*.

    Priority order:
    1. ```python … ``` fenced block
    2. ``` … ``` fenced block (language-agnostic)
    3. Bare code: if the whole response (after stripping think tags) looks like
       Python source (contains a ``def `` statement), return it as-is.

    Returns ``None`` if nothing Python-like is found.
    """
    # --- 1. ```python ... ``` ---
    pattern_python = re.compile(
        r"```python\s*\n(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern_python.search(response)
    if match:
        return match.group(1).strip()

    # --- 2. ``` ... ``` ---
    pattern_generic = re.compile(r"```\s*\n(.*?)```", re.DOTALL)
    match = pattern_generic.search(response)
    if match:
        candidate = match.group(1).strip()
        # Require at least a `def` to count as Python
        if "def " in candidate:
            return candidate

    # --- 3. Bare code (strip think block first) ---
    bare = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    if "def " in bare:
        return bare

    return None
