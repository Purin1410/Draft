"""LaTeX token tree parser used for TAMER structural supervision.

The parser is adapted from the TAMER reference implementation and returns,
for each input token, the token index of its structural parent. Tokens that do
not appear in the parsed tree receive ``-1`` and should be ignored by loss
functions. The parser is intentionally Python-list based because this is
symbolic tree construction, not tensor math.
"""

from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class Symbol:
    idx: int
    token: str

    def __eq__(self, value: object) -> bool:
        if isinstance(value, str):
            return self.token == value
        return super().__eq__(value)

    def __str__(self) -> str:
        return self.token


class Node:
    def __init__(self, x):
        self.x = x
        self.childs = []
        self.relations = []


class LatexParseError(ValueError):
    """Raised when a token sequence cannot be parsed as a supported tree."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LatexParseError(message)


def findnextbracket(latex: list, leftbracket: str = "{") -> int:
    if leftbracket == "{":
        rightbracket = "}"
    elif leftbracket == "[":
        rightbracket = "]"
    else:
        raise LatexParseError(f"Unknown bracket: {leftbracket}")

    depth = 0
    for idx, token in enumerate(latex):
        if token == leftbracket:
            depth += 1
        if token == rightbracket:
            depth -= 1
            if depth == 0:
                return idx
    return -1


def findendmatrix(latex: list) -> int:
    depth = 1
    for idx, token in enumerate(latex):
        if token == "\\begin{matrix}":
            depth += 1
        if token == "\\end{matrix}":
            depth -= 1
            if depth == 0:
                return idx
    return -1


def _pop_group(latex: list, leftbracket: str = "{") -> list:
    _require(len(latex) > 0 and latex[0] == leftbracket, f"expected {leftbracket}")
    end = findnextbracket(latex, leftbracket=leftbracket)
    _require(end >= 0, f"unclosed {leftbracket} group")
    group = latex[1:end]
    for _ in range(end + 1):
        latex.pop(0)
    return group


def _append_group_child(cur_node: Node, latex: list, relation: str, leftbracket: str = "{") -> None:
    cur_node.childs.append(latex2Tree(_pop_group(latex, leftbracket=leftbracket)))
    cur_node.relations.append(relation)


def _append_script_if_present(cur_node: Node, latex: list, token: str, relation: str) -> None:
    if len(latex) > 0 and latex[0] == token:
        latex.pop(0)
        _append_group_child(cur_node, latex, relation, "{")


def latex2Tree(latex: list) -> Node:
    """Build a structural tree from a mutable list of ``Symbol`` objects."""
    if len(latex) == 0:
        return Node("<eol>")

    cur_node = Node(latex.pop(0))
    symbol = cur_node.x

    if symbol == "<bol>":
        _append_script_if_present(cur_node, latex, "_", "sub")
        _append_script_if_present(cur_node, latex, "^", "sup")

    elif symbol == "\\begin{matrix}":
        end = findendmatrix(latex)
        _require(end >= 0, "unclosed matrix")
        cur_node.childs.append(latex2Tree(latex[:end]))
        cur_node.relations.append("Mstart")
        for _ in range(end + 1):
            latex.pop(0)

    elif symbol in {"\\iint", "\\bigcup", "\\sum", "\\lim", "\\coprod"}:
        _append_script_if_present(cur_node, latex, "_", "below")
        _append_script_if_present(cur_node, latex, "^", "above")

    elif symbol in {
        "\\dot",
        "\\ddot",
        "\\hat",
        "\\check",
        "\\grave",
        "\\acute",
        "\\tilde",
        "\\breve",
        "\\bar",
        "\\vec",
        "\\widehat",
        "\\overbrace",
        "\\widetilde",
        "\\overleftarrow",
        "\\overrightarrow",
        "\\overline",
    }:
        _append_group_child(cur_node, latex, "below", "{")

    elif symbol in {"\\underline", "\\underbrace"}:
        _append_group_child(cur_node, latex, "above", "{")

    elif symbol in {"\\xrightarrow", "\\xleftarrow"}:
        if len(latex) > 0 and latex[0] == "[":
            _append_group_child(cur_node, latex, "below", "[")
        if len(latex) > 0 and latex[0] == "{":
            _append_group_child(cur_node, latex, "above", "{")

    elif symbol == "\\frac":
        _append_group_child(cur_node, latex, "above", "{")
        below = latex2Tree(_pop_group(latex, "{"))
        cur_node.childs.insert(-1, below)
        cur_node.relations.insert(-1, "below")

    elif symbol == "\\sqrt":
        if len(latex) > 0 and latex[0] == "[":
            _append_group_child(cur_node, latex, "leftup", "[")
        _append_group_child(cur_node, latex, "inside", "{")

    else:
        _append_script_if_present(cur_node, latex, "_", "sub")
        _append_script_if_present(cur_node, latex, "^", "sup")

    if len(latex) > 0 and latex[0] == "\\\\":
        latex.pop(0)
        relation = "nextline"
    elif len(latex) > 0:
        relation = "right"
    else:
        relation = "end"

    cur_node.childs.append(latex2Tree(latex))
    cur_node.relations.append(relation)
    return cur_node


def node2list(tree: Node) -> list:
    index = 0
    gtd = []

    def _node2list(parent, parent_index: int, relation: str, current: Node, initial: bool = False):
        nonlocal index
        if current is None or current.x == "<eol>":
            return
        index = 1 if initial else index + 1
        gtd.append([current.x, index, parent, parent_index, relation])
        parent_index = index
        for child, child_relation in zip(current.childs, current.relations):
            _node2list(current.x, parent_index, child_relation, child)

    _node2list(Symbol(-1, "<sos>"), 0, "start", tree, initial=True)
    return gtd


def to_struct(latex_list: Sequence[str]) -> List[int]:
    """Return parent token indices for ``latex_list``.

    The returned list has the same length as ``latex_list``. Values are input
    token indices; ``-1`` marks ignored tokens such as grouping braces.
    """
    symbols = [Symbol(i, token) for i, token in enumerate(latex_list)]
    try:
        tree = latex2Tree(symbols)
        gtd = node2list(tree)
    except (AssertionError, IndexError) as exc:
        raise LatexParseError(str(exc)) from exc

    parents = {node[0].idx: node[2].idx for node in gtd if isinstance(node[0], Symbol)}
    return [parents.get(i, -1) for i in range(len(latex_list))]


def safe_to_struct(latex_list: Sequence[str]) -> tuple[List[int], bool]:
    """Parse tokens and return ``(parents, illegal)`` without raising."""
    try:
        return to_struct(latex_list), False
    except (LatexParseError, IndexError, TypeError):
        return [-1 for _ in latex_list], True
