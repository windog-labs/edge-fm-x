"""Audit exported TensorRegions for hidden mutation, RNG, and I/O."""

from __future__ import annotations

import inspect
import operator
from types import ModuleType
from typing import Any, Callable

from vlaforge.deployment.contract import (
    ArtifactDiagnostic,
    DiagnosticSeverity,
    EffectAudit,
)


_RANDOM_TOKENS = (
    "rand",
    "bernoulli",
    "dropout",
    "multinomial",
    "normal",
    "poisson",
)
_EXTERNAL_IO_TOKENS = (
    "read",
    "write",
    "socket",
    "print",
    "open",
)


def audit_callable_closure(function: Callable[..., object]) -> tuple[
    ArtifactDiagnostic, ...
]:
    """Reject mutable/non-serializable values captured by a plain function."""

    try:
        closure = inspect.getclosurevars(function)
    except (TypeError, ValueError):
        return ()
    diagnostics: list[ArtifactDiagnostic] = []
    for name, value in sorted(closure.nonlocals.items()):
        if _is_immutable_literal(value):
            continue
        diagnostics.append(
            ArtifactDiagnostic(
                code="frontend.nonserializable_closure",
                message=(
                    f"closure variable {name!r} has unsupported mutable type "
                    f"{type(value).__qualname__}"
                ),
                source=getattr(function, "__qualname__", repr(function)),
            )
        )
    return tuple(diagnostics)


def audit_exported_program(
    exported_program: Any,
    *,
    closure_diagnostics: tuple[ArtifactDiagnostic, ...] = (),
    explicit_rng: bool = False,
    lifted_states: tuple[str, ...] = (),
) -> EffectAudit:
    diagnostics = list(closure_diagnostics)
    hidden_mutation = False
    hidden_rng = False
    external_io = False
    local_mutations = 0
    deterministic_dropouts = 0

    for node in exported_program.graph_module.graph.nodes:
        if node.op not in {"call_function", "call_method", "call_module"}:
            continue
        target = node.target
        target_name = str(target).lower()
        schema = getattr(target, "_schema", None)
        if schema is not None and bool(getattr(schema, "is_mutable", False)):
            written_values = tuple(
                node.args[index]
                for index, argument in enumerate(schema.arguments)
                if index < len(node.args)
                and argument.alias_info is not None
                and argument.alias_info.is_write
            )
            if any(_aliases_external_storage(value, {}) for value in written_values):
                hidden_mutation = True
                diagnostics.append(
                    ArtifactDiagnostic(
                        "frontend.hidden_mutation",
                        f"captured operator {target} mutates an input or module value",
                        source=node.name,
                    )
                )
            else:
                local_mutations += 1
        if "dropout" in target_name and _dropout_training_is_false(node, schema):
            deterministic_dropouts += 1
        elif any(token in target_name for token in _RANDOM_TOKENS):
            hidden_rng = True
            diagnostics.append(
                ArtifactDiagnostic(
                    "frontend.hidden_rng",
                    f"captured random operator {target}",
                    source=node.name,
                )
            )
        if node.op != "call_function" and any(
            token in target_name for token in _EXTERNAL_IO_TOKENS
        ):
            external_io = True
            diagnostics.append(
                ArtifactDiagnostic(
                    "frontend.external_io",
                    f"captured possible external I/O target {target}",
                    source=node.name,
                )
            )

    if local_mutations:
        diagnostics.append(
            ArtifactDiagnostic(
                "frontend.local_workspace_mutation",
                f"{local_mutations} invocation-local mutable operators do not "
                "alias inputs or module values",
                DiagnosticSeverity.INFO,
            )
        )

    if deterministic_dropouts:
        diagnostics.append(
            ArtifactDiagnostic(
                "frontend.eval_dropout",
                f"{deterministic_dropouts} dropout operators have train=false "
                "and are deterministic",
                DiagnosticSeverity.INFO,
            )
        )

    input_specs = getattr(exported_program.graph_signature, "input_specs", ())
    for spec in input_specs:
        kind = str(getattr(spec, "kind", ""))
        if "CUSTOM_OBJ" in kind.upper():
            diagnostics.append(
                ArtifactDiagnostic(
                    "frontend.custom_object",
                    f"captured custom-object input {getattr(spec, 'target', None)!r}",
                )
            )

    return EffectAudit(
        hidden_mutation=hidden_mutation,
        hidden_rng=hidden_rng,
        external_io=external_io,
        explicit_rng=explicit_rng,
        lifted_states=lifted_states,
        diagnostics=tuple(diagnostics),
    )


def _is_immutable_literal(value: object) -> bool:
    if value is None or isinstance(value, str | bytes | int | float | bool):
        return True
    if isinstance(value, tuple):
        return all(_is_immutable_literal(item) for item in value)
    if isinstance(value, frozenset):
        return all(_is_immutable_literal(item) for item in value)
    if isinstance(value, ModuleType):
        return True
    return False


def _aliases_external_storage(value: object, memo: dict[object, bool]) -> bool:
    """Conservatively follow only schema-declared alias-preserving edges."""

    try:
        from torch.fx import Node
    except ImportError:
        return False
    if not isinstance(value, Node):
        return False
    if value in memo:
        return memo[value]
    if value.op in {"placeholder", "get_attr"}:
        memo[value] = True
        return True
    if value.op == "call_function" and value.target is operator.getitem:
        result = _aliases_external_storage(value.args[0], memo)
        memo[value] = result
        return result
    schema = getattr(value.target, "_schema", None)
    if schema is None:
        memo[value] = False
        return False
    return_aliases = [
        item.alias_info for item in schema.returns if item.alias_info is not None
    ]
    if not return_aliases:
        memo[value] = False
        return False
    alias_sets = set().union(
        *(
            set(alias.before_set) | set(alias.after_set)
            for alias in return_aliases
        )
    )
    result = False
    for index, argument in enumerate(schema.arguments):
        alias = argument.alias_info
        if alias is None or index >= len(value.args):
            continue
        argument_sets = set(alias.before_set) | set(alias.after_set)
        if alias_sets & argument_sets:
            result |= _aliases_external_storage(value.args[index], memo)
    memo[value] = result
    return result


def _dropout_training_is_false(node: object, schema: object | None) -> bool:
    if schema is None:
        return False
    arguments = getattr(schema, "arguments", ())
    for index, argument in enumerate(arguments):
        if getattr(argument, "name", None) not in {"train", "training"}:
            continue
        if index < len(node.args):
            value = node.args[index]
        else:
            value = node.kwargs.get(argument.name)
        return value is False
    return False
