# Created by Oliver Meihls

# Delphi tool plane with schema validation, RBAC, and Zeus governance hooks.

from __future__ import annotations

import importlib
import inspect
import pkgutil
from dataclasses import dataclass, field
from enum import Enum
from types import UnionType
from typing import Any, Callable, Dict, Mapping, Optional, Union, get_args, get_origin, get_type_hints

from src.agent.zeus import ZeusPolicyEngine


class RiskLevel(str, Enum):
    # Risk category attached to a tool definition.

    READ_ONLY = "READ_ONLY"
    LOW = "LOW"
    HIGH = "HIGH"


@dataclass(frozen=True)
class ToolDefinition:
    # Resolved metadata for a registered tool function.

    name: str
    description: str
    risk_level: RiskLevel
    parameters_schema: Optional[Dict[str, Any]]
    func: Callable[..., Any]
    estimated_cost: float = 0.0


@dataclass
class ToolResult:
    # Structured response for all Delphi tool calls.

    success: bool
    tool_name: str
    data: Any = None
    error: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class PendingToolDefinition:
    # Decorator-attached metadata used by discovery-based registration.

    name: str
    description: str
    risk_level: RiskLevel
    parameters_schema: Optional[Dict[str, Any]]
    estimated_cost: float = 0.0


def tool(
    *,
    name: str,
    description: str,
    risk_level: RiskLevel,
    parameters_schema: Optional[Dict[str, Any]] = None,
    estimated_cost: float = 0.0,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    # Mark a function as a Delphi tool for later discovery/registration.

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        setattr(
            func,
            "__delphi_tool_definition__",
            PendingToolDefinition(
                name=name,
                description=description,
                risk_level=risk_level,
                parameters_schema=parameters_schema,
                estimated_cost=estimated_cost,
            ),
        )
        return func

    return decorator


class DelphiToolRegistry:
    # Allowlisted tool registry and execution gateway for Pantheon agents.

    def __init__(
        self,
        zeus: ZeusPolicyEngine,
        role_tool_allowlist: Optional[Mapping[str, set[str]]] = None,
    ):
        self.zeus = zeus
        self._tools: Dict[str, ToolDefinition] = {}
        self._role_tool_allowlist: Dict[str, set[str]] = {
            role: set(tools) for role, tools in (role_tool_allowlist or {}).items()
        }

    def register(
        self,
        *,
        name: str,
        description: str,
        risk_level: RiskLevel,
        parameters_schema: Optional[Dict[str, Any]] = None,
        estimated_cost: float = 0.0,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        # Decorator that registers a function as an executable Delphi tool.

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._register_tool(
                ToolDefinition(
                    name=name,
                    description=description,
                    risk_level=risk_level,
                    parameters_schema=parameters_schema,
                    func=func,
                    estimated_cost=estimated_cost,
                )
            )
            return func

        return decorator

    def discover_tools(self, package_name: str = "src") -> None:
        # Import package modules and register any @tool-annotated callables.

        package = importlib.import_module(package_name)
        paths = getattr(package, "__path__", None)
        if not paths:
            return

        for module_info in pkgutil.walk_packages(paths, prefix=f"{package_name}."):
            module = importlib.import_module(module_info.name)
            for _, member in inspect.getmembers(module, inspect.isfunction):
                pending = getattr(member, "__delphi_tool_definition__", None)
                if pending is None:
                    continue
                self._register_tool(
                    ToolDefinition(
                        name=pending.name,
                        description=pending.description,
                        risk_level=pending.risk_level,
                        parameters_schema=pending.parameters_schema,
                        func=member,
                        estimated_cost=pending.estimated_cost,
                    )
                )

    async def call_tool(self, tool_name: str, args: dict, actor: str) -> ToolResult:
        # Validate, authorize, execute, and audit a tool call.

        metadata: Dict[str, Any] = {
            "event": "tool_attempt",
            "actor": actor,
            "tool_name": tool_name,
            "args": args,
        }

        tool_def = self._tools.get(tool_name)
        if tool_def is None:
            result = self._error_result(tool_name, "TOOL_NOT_FOUND", "Tool is not registered.")
            await self.zeus.log_action({**metadata, "status": "failure", "error": result.error})
            return result

        if not self._is_role_allowed(actor, tool_name):
            result = self._error_result(tool_name, "RBAC_DENIED", f"Actor '{actor}' is not allowed to call '{tool_name}'.")
            await self.zeus.log_action({**metadata, "status": "failure", "error": result.error})
            return result

        validation_error = self._validate_args(tool_def, args)
        if validation_error is not None:
            result = self._error_result(tool_name, "VALIDATION_ERROR", validation_error)
            await self.zeus.log_action({**metadata, "status": "failure", "error": result.error})
            return result

        if tool_def.estimated_cost > 0 and not self.zeus.check_budget(tool_def.estimated_cost):
            result = self._error_result(tool_name, "BUDGET_DENIED", "Budget policy denied this tool call.")
            await self.zeus.log_action({**metadata, "status": "failure", "error": result.error})
            return result

        if tool_def.risk_level == RiskLevel.HIGH and self.zeus.is_approval_required(tool_name):
            result = self._error_result(tool_name, "APPROVAL_REQUIRED", "Operator approval required by Zeus policy.")
            await self.zeus.log_action({**metadata, "status": "failure", "error": result.error})
            return result

        try:
            outcome = tool_def.func(**args)
            if inspect.isawaitable(outcome):
                outcome = await outcome
            result = ToolResult(success=True, tool_name=tool_name, data=outcome)
            await self.zeus.log_action({**metadata, "status": "success"})
            return result
        except Exception as exc:  # pragma: no cover - defensive boundary
            result = self._error_result(tool_name, "TOOL_EXECUTION_ERROR", str(exc))
            await self.zeus.log_action({**metadata, "status": "failure", "error": result.error})
            return result

    def _register_tool(self, tool_def: ToolDefinition) -> None:
        if tool_def.name in self._tools:
            import logging
            logging.getLogger("argus.delphi").warning(
                f"Tool '{tool_def.name}' is already registered. Skipping duplicate registration."
            )
            return
        self._tools[tool_def.name] = tool_def

    def _is_role_allowed(self, actor: str, tool_name: str) -> bool:
        allowed_tools = self._role_tool_allowlist.get(actor)
        if allowed_tools is None:
            return False
        return "*" in allowed_tools or tool_name in allowed_tools

    def _validate_args(self, tool_def: ToolDefinition, args: dict) -> Optional[str]:
        if not isinstance(args, dict):
            return "Tool arguments must be a JSON object."

        if tool_def.parameters_schema:
            return self._validate_against_schema(tool_def.parameters_schema, args)
        return self._validate_against_signature(tool_def.func, args)

    def _validate_against_schema(self, schema: Dict[str, Any], args: Dict[str, Any]) -> Optional[str]:
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        allowed_keys = set(properties.keys())

        unknown = sorted(set(args.keys()) - allowed_keys)
        if unknown:
            return f"Unexpected argument(s): {', '.join(unknown)}."

        missing = sorted(required - set(args.keys()))
        if missing:
            return f"Missing required argument(s): {', '.join(missing)}."

        for key, value in args.items():
            expected = properties.get(key, {}).get("type")
            if expected and not self._matches_json_type(value, expected):
                return f"Argument '{key}' must be of type '{expected}'."
        return None

    def _validate_against_signature(self, func: Callable[..., Any], args: Dict[str, Any]) -> Optional[str]:
        signature = inspect.signature(func)
        hints = get_type_hints(func)

        required = {
            name
            for name, param in signature.parameters.items()
            if param.default is inspect._empty and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        allowed = {
            name
            for name, param in signature.parameters.items()
            if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }

        unknown = sorted(set(args.keys()) - allowed)
        if unknown:
            return f"Unexpected argument(s): {', '.join(unknown)}."

        missing = sorted(required - set(args.keys()))
        if missing:
            return f"Missing required argument(s): {', '.join(missing)}."

        for key, value in args.items():
            if key in hints and not self._matches_python_type(value, hints[key]):
                return f"Argument '{key}' has invalid type."

        return None

    def _matches_json_type(self, value: Any, expected: str) -> bool:
        mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "object": dict,
            "array": list,
        }
        expected_type = mapping.get(expected)
        if expected_type is None:
            return True
        if expected == "integer" and isinstance(value, bool):
            return False
        return isinstance(value, expected_type)

    def _matches_python_type(self, value: Any, annotation: Any) -> bool:
        origin = get_origin(annotation)
        if origin is None:
            if annotation is Any:
                return True
            if annotation is int and isinstance(value, bool):
                return False
            return isinstance(value, annotation)

        if origin is list:
            (inner_type,) = get_args(annotation) or (Any,)
            return isinstance(value, list) and all(self._matches_python_type(item, inner_type) for item in value)

        if origin is dict:
            return isinstance(value, dict)

        if origin is tuple:
            return isinstance(value, tuple)

        if origin in (Union, UnionType):
            return any(self._matches_python_type(value, option) for option in get_args(annotation))

        return True

    def _error_result(self, tool_name: str, code: str, message: str) -> ToolResult:
        return ToolResult(success=False, tool_name=tool_name, error={"code": code, "message": message})

    @property
    def tools(self) -> Dict[str, ToolDefinition]:
        return dict(self._tools)
