# jax2onnx/converter/state_inputs.py

"""Expose selected captured arrays as named graph inputs with saved defaults."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, MutableSequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import onnx_ir as ir


@dataclass(frozen=True)
class StateInput:
    """Export one captured array as a named ONNX input with a saved default.

    Attributes:
        name: Name of the graph input and of its default tensor.
        metadata: String properties stored on the default tensor.
        overridable: Keep the array in ``graph.input`` so callers can replace it.
            When false it is an ordinary initializer, which runtimes may fold.
    """

    name: str
    metadata: Mapping[str, str] = field(default_factory=dict)
    overridable: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(
                f"StateInput name must be a nonempty string, got {self.name!r}."
            )
        for key, value in self.metadata.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError(
                    f"StateInput {self.name!r} metadata must map strings to strings."
                )


StateSelector = Callable[[object], StateInput | None]


@dataclass(frozen=True)
class _SelectedArray:
    # Holding the array keeps its id unique for the rest of the export.
    array: object
    spec: StateInput
    default: ir.TensorProtocol


class StateInputs:
    """The arrays one export selected; every scope of that export shares this."""

    def __init__(
        self,
        select: StateSelector,
        *,
        graph_inputs: MutableSequence[ir.Value],
        reserved_names: Collection[str],
    ) -> None:
        self._select = select
        self._graph_inputs = graph_inputs
        self._reserved_names = reserved_names
        self._selected: dict[int, _SelectedArray] = {}

    def input_name(self, var: Any, array: object) -> str | None:
        """Return the input name of ``array``, or ``None`` to keep it constant.

        The first selection of an array object adds its root graph input, so
        every capture of that object reads the same input.
        """
        selected = self._selected.get(id(array))
        if selected is not None:
            return selected.spec.name
        spec = self._select(array)
        if spec is None:
            return None
        if not isinstance(spec, StateInput):
            raise TypeError(
                "The state callback must return StateInput or None, "
                f"got {type(spec).__name__}."
            )
        self._add_input(var, array, spec)
        return spec.name

    def _add_input(self, var: Any, array: object, spec: StateInput) -> None:
        taken = {value.name for value in self._graph_inputs}
        if spec.name in taken or spec.name in self._reserved_names:
            raise ValueError(f"State input name {spec.name!r} is already in use.")
        payload = np.asarray(array, dtype=var.aval.dtype)
        if np.iscomplexobj(payload):
            raise TypeError(f"State input {spec.name!r} cannot be complex.")
        default = ir.tensor(payload, name=spec.name)
        default.metadata_props.update(spec.metadata)
        self._graph_inputs.append(
            ir.Value(
                name=spec.name,
                type=ir.TensorType(default.dtype),
                shape=ir.Shape(payload.shape),
            )
        )
        self._selected[id(array)] = _SelectedArray(array, spec, default)

    def attach_defaults(self, model: ir.Model) -> None:
        """Store each default on its input once graph rewriting has finished.

        Attaching last means the optimizer sees an ordinary input it cannot fold
        or copy. An input missing here was pruned because nothing reads it.
        """
        graph = model.graph
        inputs = {value.name: value for value in graph.inputs}
        for selected in self._selected.values():
            value = inputs.get(selected.spec.name)
            if value is None:
                continue
            value.const_value = selected.default
            graph.register_initializer(value)
            if not selected.spec.overridable:
                graph.inputs.remove(value)
