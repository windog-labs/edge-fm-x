"""Typed builders for core VLA operations.

Operation names are deliberately model-independent. Model-specific behavior is
isolated behind ``vla.invoke`` tensor regions.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from vlaforge.ir.attrs import EpochExpr
from vlaforge.ir.program import Block, Operation, Value
from vlaforge.ir.types import (
    ActionType,
    CommittedActionType,
    EpochType,
    EventType,
    FutureType,
    IRType,
    PendingType,
    ScalarType,
    SnapshotType,
    TransactionType,
)


def op(
    opcode: str,
    *,
    results: Iterable[Value] = (),
    operands: Iterable[str] = (),
    attributes: Mapping[str, Any] | None = None,
    regions: Iterable[Block] = (),
    location: str | None = None,
) -> Operation:
    return Operation(
        opcode,
        tuple(results),
        tuple(operands),
        attributes or {},
        tuple(regions),
        location,
    )


def sample_input(
    value_name: str,
    epoch_name: str,
    payload: IRType,
    stream: str,
    clock: str,
    *,
    max_age_ns: int | None = None,
) -> Operation:
    attrs: dict[str, Any] = {"stream": stream}
    if max_age_ns is not None:
        attrs["max_age_ns"] = max_age_ns
    return op(
        "vla.sample_input",
        results=(Value(value_name, payload), Value(epoch_name, EpochType(clock))),
        attributes=attrs,
    )


def transaction_begin(name: str, epoch: str) -> Operation:
    return op(
        "vla.txn.begin",
        results=(Value(name, TransactionType()),),
        operands=(epoch,),
    )


def state_read(
    name: str,
    payload: IRType,
    state: str,
    transaction: str,
    *,
    epoch: EpochExpr,
    version: str = "latest",
) -> Operation:
    return op(
        "vla.state.read",
        results=(Value(name, SnapshotType(state, payload)),),
        operands=(transaction,),
        attributes={
            "state": state,
            "epoch": epoch.to_dict(),
            "version": version,
        },
    )


def snapshot_value(
    name: str,
    payload: IRType,
    snapshot: str,
) -> Operation:
    return op(
        "vla.snapshot.value",
        results=(Value(name, payload),),
        operands=(snapshot,),
    )


def invoke(
    result_names: Iterable[str],
    result_types: Iterable[IRType],
    region: str,
    operands: Iterable[str],
) -> Operation:
    results = tuple(
        Value(name, result_type)
        for name, result_type in zip(result_names, result_types, strict=True)
    )
    return op(
        "vla.invoke",
        results=results,
        operands=operands,
        attributes={"region": region},
    )


def for_loop(
    result: Value,
    initial: str,
    induction: Value,
    iter_arg: Value,
    body: Block,
    *,
    lower: int,
    upper: int,
    step: int = 1,
) -> Operation:
    return op(
        "vla.for",
        results=(result,),
        operands=(initial,),
        attributes={
            "induction": induction.name,
            "iter_arg": iter_arg.name,
            "lower": lower,
            "upper": upper,
            "step": step,
        },
        regions=(Block((induction, iter_arg), body.operations),),
    )


def while_loop(
    results: Iterable[Value],
    operands: Iterable[str],
    condition: Block,
    body: Block,
    *,
    max_iterations: int,
) -> Operation:
    return op(
        "vla.while",
        results=results,
        operands=operands,
        attributes={"max_iterations": max_iterations},
        regions=(condition, body),
    )


def if_op(
    results: Iterable[Value],
    condition: str,
    then_block: Block,
    else_block: Block,
) -> Operation:
    return op(
        "vla.if",
        results=results,
        operands=(condition,),
        regions=(then_block, else_block),
    )


def yield_values(*operands: str) -> Operation:
    return op("vla.yield", operands=operands)


def return_values(*operands: str) -> Operation:
    return op("vla.return", operands=operands)


def stage_write(
    pending_name: str,
    payload: IRType,
    state: str,
    transaction: str,
    value: str,
    *,
    epoch: EpochExpr,
    inplace: bool = False,
) -> Operation:
    return op(
        "vla.state.stage_write",
        results=(Value(pending_name, PendingType(state, payload)),),
        operands=(transaction, value),
        attributes={
            "state": state,
            "epoch": epoch.to_dict(),
            "inplace": inplace,
        },
    )


def validate(name: str, value: str, contract: str) -> Operation:
    return op(
        "vla.validate",
        results=(Value(name, ScalarType("bool")),),
        operands=(value,),
        attributes={"contract": contract},
    )


def action_create(name: str, value: str, payload: IRType, epoch: str) -> Operation:
    return op(
        "vla.action.create",
        results=(Value(name, ActionType(payload)),),
        operands=(value, epoch),
    )


def transaction_commit(
    committed_name: str,
    payload: IRType,
    transaction: str,
    action: str,
    condition: str,
    *,
    required_futures: Iterable[str] = (),
) -> Operation:
    return op(
        "vla.txn.commit",
        results=(Value(committed_name, CommittedActionType(payload)),),
        operands=(transaction, action, condition),
        attributes={"required_futures": list(required_futures)},
    )


def transaction_abort(transaction: str, *, reason: str = "") -> Operation:
    return op(
        "vla.txn.abort",
        operands=(transaction,),
        attributes={"reason": reason},
    )


def action_publish(committed_action: str) -> Operation:
    return op("vla.action.publish", operands=(committed_action,))


def reset(*states: str) -> Operation:
    return op("vla.reset", attributes={"states": list(states)})


def async_execute(
    future_name: str,
    payload: IRType,
    body: Block,
    *,
    reads: Iterable[str] = (),
    writes: Iterable[str] = (),
) -> Operation:
    return op(
        "vla.async",
        results=(Value(future_name, FutureType(payload)), Value(f"{future_name}_event", EventType())),
        attributes={"reads": list(reads), "writes": list(writes)},
        regions=(body,),
    )


def await_future(name: str, payload: IRType, future: str) -> Operation:
    return op(
        "vla.await",
        results=(Value(name, payload),),
        operands=(future,),
    )
