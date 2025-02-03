from typing import TypeVar

actionKey = TypeVar("actionKey", str, int)
solverKey = TypeVar("solverKey", str, int)
agentKey = TypeVar("agentKey", str, int)


__all__: list[str] = ["actionKey", "solverKey", "agentKey"]
