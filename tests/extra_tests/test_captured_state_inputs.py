# tests/extra_tests/test_captured_state_inputs.py

"""``to_onnx(state=...)`` exposes captured arrays as named inputs with defaults."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import equinox as eqx
from flax import linen, nnx
import jax
import jax.numpy as jnp
import numpy as np
import onnx
from onnx import numpy_helper
import onnx_ir as ir
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from jax2onnx import StateInput, allclose, onnx_function, to_onnx

_DATA: np.ndarray = np.array([1.5, -0.5], dtype=np.float32)
_INPUT_SPEC: list[jax.ShapeDtypeStruct] = [jax.ShapeDtypeStruct((2,), jnp.float32)]


def _weight() -> jax.Array:
    return jnp.asarray([2.0, 3.0], dtype=jnp.float32)


def _select(named: Mapping[str, object], **options: Any) -> Callable[..., Any]:
    specs = {id(array): StateInput(name, **options) for name, array in named.items()}
    return lambda array: specs.get(id(array))


def _session(model: onnx.ModelProto) -> ort.InferenceSession:
    return ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )


def _run(model: onnx.ModelProto, feed: Mapping[str, Any]) -> list[np.ndarray]:
    arrays = {name: np.asarray(value) for name, value in feed.items()}
    return _session(model).run(None, arrays)


def _input_names(model: onnx.ModelProto) -> list[str]:
    return [value.name for value in model.graph.input]


def _initializer(model: onnx.ModelProto, name: str) -> onnx.TensorProto:
    return next(tensor for tensor in model.graph.initializer if tensor.name == name)


def _noisy(value: Any, seed: int) -> jax.Array:
    array = np.asarray(value)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.7, 0.2, array.shape).astype(array.dtype)
    return jnp.asarray(array + noise)


def _closure(x: jax.Array, w: jax.Array) -> jax.Array:
    return x * w


def _nested_jit(x: jax.Array, w: jax.Array) -> jax.Array:
    return jax.jit(lambda y: y * w)(x)


def _doubly_nested_jit(x: jax.Array, w: jax.Array) -> jax.Array:
    return jax.jit(lambda y: jax.jit(lambda z: z * w)(y))(x)


def _checkpoint(x: jax.Array, w: jax.Array) -> jax.Array:
    return jax.checkpoint(lambda y: y * w)(x)


def _custom_jvp(x: jax.Array, w: jax.Array) -> jax.Array:
    @jax.custom_jvp
    def multiply(y: jax.Array) -> jax.Array:
        return y * w

    multiply.defjvp(lambda primals, tangents: (primals[0] * w, tangents[0] * w))
    return multiply(x)


def _custom_vjp(x: jax.Array, w: jax.Array) -> jax.Array:
    @jax.custom_vjp
    def multiply(y: jax.Array) -> jax.Array:
        return y * w

    multiply.defvjp(lambda y: (y * w, None), lambda _, cotangent: (cotangent * w,))
    return multiply(x)


def _cond(x: jax.Array, w: jax.Array) -> jax.Array:
    return jax.lax.cond(jnp.sum(x) > 0, lambda y: y * w, lambda y: y + w, x)


def _cond_nested_jit(x: jax.Array, w: jax.Array) -> jax.Array:
    return jax.lax.cond(
        jnp.sum(x) > 0, lambda y: jax.jit(lambda z: z * w)(y), lambda y: y + 1.0, x
    )


def _scan(x: jax.Array, w: jax.Array) -> jax.Array:
    carry, _ = jax.lax.scan(lambda c, _: (c * w, None), x, None, length=2)
    return carry


def _scan_nested_jit(x: jax.Array, w: jax.Array) -> jax.Array:
    def body(carry: jax.Array, _: None) -> tuple[jax.Array, None]:
        return jax.jit(lambda y: y * w)(carry), None

    carry, _ = jax.lax.scan(body, x, None, length=2)
    return carry


def _scan_xs_nested_jit(x: jax.Array, w: jax.Array) -> jax.Array:
    def body(carry: jax.Array, element: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jax.jit(lambda y: y * w)(carry) + element, element

    carry, _ = jax.lax.scan(body, x, jnp.stack([x, x]))
    return carry


def _while(x: jax.Array, w: jax.Array) -> jax.Array:
    state = (x, jnp.asarray(0))
    return jax.lax.while_loop(
        lambda s: s[1] < 2, lambda s: (s[0] * w, s[1] + 1), state
    )[0]


def _while_nested_jit(x: jax.Array, w: jax.Array) -> jax.Array:
    def body(state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        return jax.jit(lambda y: y * w)(state[0]), state[1] + 1

    return jax.lax.while_loop(lambda s: s[1] < 2, body, (x, jnp.asarray(0)))[0]


def _while_predicate_nested_jit(x: jax.Array, w: jax.Array) -> jax.Array:
    def predicate(state: tuple[jax.Array, jax.Array]) -> jax.Array:
        below_limit = jax.jit(lambda y: jnp.sum(y * w) < 1e3)(state[0])
        return jnp.logical_and(below_limit, state[1] < 2)

    state = (x, jnp.asarray(0))
    return jax.lax.while_loop(predicate, lambda s: (s[0] + 1.0, s[1] + 1), state)[0]


def _fori_loop(x: jax.Array, w: jax.Array) -> jax.Array:
    return jax.lax.fori_loop(0, 2, lambda _, carry: carry * w, x)


def _jit_inside_cond_inside_scan(x: jax.Array, w: jax.Array) -> jax.Array:
    def body(carry: jax.Array, _: None) -> tuple[jax.Array, None]:
        multiplied = jax.lax.cond(
            jnp.sum(carry) > 0,
            lambda y: jax.jit(lambda z: z * w)(y),
            lambda y: y - 1.0,
            carry,
        )
        return multiplied, None

    carry, _ = jax.lax.scan(body, x, None, length=2)
    return carry


@pytest.mark.parametrize(
    "computation",
    [
        _closure,
        _nested_jit,
        _doubly_nested_jit,
        _checkpoint,
        _custom_jvp,
        _custom_vjp,
        _cond,
        _cond_nested_jit,
        _scan,
        _scan_nested_jit,
        _scan_xs_nested_jit,
        _while,
        _while_nested_jit,
        _while_predicate_nested_jit,
        _fori_loop,
        _jit_inside_cond_inside_scan,
    ],
    ids=lambda computation: computation.__name__.strip("_"),
)
def test_every_scope_reads_the_selected_input(computation: Callable[..., Any]) -> None:
    w = _weight()

    def fn(x: jax.Array) -> jax.Array:
        return computation(x, w)

    model = to_onnx(fn, inputs=_INPUT_SPEC, opset=21, state=_select({"w": w}))
    onnx.checker.check_model(model)

    assert _input_names(model) == ["in_0", "w"]
    assert np.array_equal(numpy_helper.to_array(_initializer(model, "w")), w)
    np.testing.assert_allclose(
        _run(model, {"in_0": _DATA})[0], computation(jnp.asarray(_DATA), w), rtol=1e-5
    )
    override = jnp.asarray([0.5, 4.0], dtype=jnp.float32)
    np.testing.assert_allclose(
        _run(model, {"in_0": _DATA, "w": override})[0],
        computation(jnp.asarray(_DATA), override),
        rtol=1e-5,
    )


def _assert_leaves_become_inputs(
    fn: Callable[..., Any],
    rebuild: Callable[[list[Any]], Callable[..., Any]],
    leaves: list[Any],
    data: np.ndarray,
) -> None:
    named = {f"state_{index}": leaf for index, leaf in enumerate(leaves)}
    model = to_onnx(
        fn,
        inputs=[jax.ShapeDtypeStruct(data.shape, data.dtype)],
        input_names=["data"],
        opset=21,
        state=_select(named),
    )
    onnx.checker.check_model(model)

    assert _input_names(model)[0] == "data"
    assert sorted(_input_names(model)[1:]) == sorted(named)
    feed = {"data": data}
    np.testing.assert_allclose(
        _run(model, feed)[0], fn(jnp.asarray(data)), rtol=1e-4, atol=1e-5
    )
    replaced = [_noisy(leaf, seed) for seed, leaf in enumerate(leaves)]
    overrides = {name: value for name, value in zip(named, replaced)}
    np.testing.assert_allclose(
        _run(model, {**feed, **overrides})[0],
        rebuild(replaced)(jnp.asarray(data)),
        rtol=1e-3,
        atol=1e-4,
    )


def _equinox_linear() -> eqx.Module:
    return eqx.nn.Linear(3, 2, key=jax.random.PRNGKey(0))


class _EquinoxConv(eqx.Module):
    conv: eqx.nn.Conv2d

    def __init__(self) -> None:
        self.conv = eqx.nn.Conv2d(3, 2, 1, key=jax.random.PRNGKey(0))

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.conv(x[:, None, None])[:, 0, 0]


class _EquinoxLayerNormed(eqx.Module):
    linear: eqx.nn.Linear
    norm: eqx.nn.LayerNorm

    def __init__(self) -> None:
        self.linear = eqx.nn.Linear(3, 3, key=jax.random.PRNGKey(0))
        self.norm = eqx.nn.LayerNorm(3)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.norm(self.linear(x))


@pytest.mark.parametrize(
    "build",
    [_equinox_linear, _EquinoxConv, _EquinoxLayerNormed],
    ids=["linear", "conv", "layer_norm"],
)
def test_equinox_leaves_become_inputs(build: Callable[[], Any]) -> None:
    model = build()
    arrays, static = eqx.partition(model, eqx.is_inexact_array)
    leaves, treedef = jax.tree_util.tree_flatten(arrays)

    def rebuild(replaced: list[Any]) -> Any:
        return eqx.combine(jax.tree_util.tree_unflatten(treedef, replaced), static)

    data = np.arange(3, dtype=np.float32) + 1.0
    _assert_leaves_become_inputs(model, rebuild, leaves, data)


def _nnx_linear() -> tuple[Any, np.ndarray]:
    return nnx.Linear(3, 2, rngs=nnx.Rngs(0)), np.ones((1, 3), dtype=np.float32)


def _nnx_linear_general() -> tuple[Any, np.ndarray]:
    module = nnx.LinearGeneral(3, (2, 2), rngs=nnx.Rngs(0))
    return module, np.ones((1, 5, 3), dtype=np.float32)


def _nnx_conv() -> tuple[Any, np.ndarray]:
    module = nnx.Conv(2, 3, kernel_size=(1, 1), rngs=nnx.Rngs(0))
    return module, np.ones((1, 2, 2, 2), dtype=np.float32)


def _nnx_batch_norm() -> tuple[Any, np.ndarray]:
    module = nnx.BatchNorm(3, use_running_average=True, rngs=nnx.Rngs(0))
    return module, np.ones((2, 3), dtype=np.float32)


def _nnx_layer_norm() -> tuple[Any, np.ndarray]:
    return nnx.LayerNorm(3, rngs=nnx.Rngs(0)), np.ones((2, 3), dtype=np.float32)


@pytest.mark.parametrize(
    "build",
    [_nnx_linear, _nnx_linear_general, _nnx_conv, _nnx_batch_norm, _nnx_layer_norm],
    ids=lambda build: build.__name__.strip("_"),
)
def test_flax_nnx_leaves_become_inputs(
    build: Callable[[], tuple[Any, np.ndarray]],
) -> None:
    module, data = build()
    graphdef, state = nnx.split(module)
    leaves, treedef = jax.tree_util.tree_flatten(state)

    def rebuild(replaced: list[Any]) -> Any:
        return nnx.merge(graphdef, jax.tree_util.tree_unflatten(treedef, replaced))

    _assert_leaves_become_inputs(module, rebuild, leaves, data)


def test_flax_linen_variables_become_inputs() -> None:
    module = linen.Dense(2)
    data = np.ones((1, 3), dtype=np.float32)
    variables = module.init(jax.random.PRNGKey(0), data)
    leaves, treedef = jax.tree_util.tree_flatten(variables)

    def rebuild(replaced: list[Any]) -> Callable[[jax.Array], jax.Array]:
        return lambda x: module.apply(
            jax.tree_util.tree_unflatten(treedef, replaced), x
        )

    _assert_leaves_become_inputs(rebuild(leaves), rebuild, leaves, data)


def test_shared_array_is_one_input() -> None:
    w = _weight()

    def fn(x: jax.Array) -> jax.Array:
        return jax.jit(lambda y: y * w)(x) + x * w

    model = to_onnx(fn, inputs=_INPUT_SPEC, opset=21, state=_select({"w": w}))

    assert _input_names(model) == ["in_0", "w"]
    assert [tensor.name for tensor in model.graph.initializer] == ["w"]


def test_equal_valued_arrays_stay_separate_inputs() -> None:
    first = _weight()
    second = jnp.array(first, copy=True)

    def fn(x: jax.Array) -> jax.Array:
        return x * first + x * second

    model = to_onnx(
        fn,
        inputs=_INPUT_SPEC,
        opset=21,
        state=_select({"first": first, "second": second}),
    )

    assert _input_names(model) == ["in_0", "first", "second"]
    doubled_first = np.asarray(first) * 2
    np.testing.assert_allclose(
        _run(model, {"in_0": _DATA, "first": doubled_first})[0],
        _DATA * doubled_first + _DATA * np.asarray(second),
    )


def test_unselected_capture_stays_a_constant() -> None:
    selected = _weight()
    constant = jnp.asarray([5.0, 7.0], dtype=jnp.float32)

    def fn(x: jax.Array) -> jax.Array:
        return x * selected + constant

    model = to_onnx(fn, inputs=_INPUT_SPEC, opset=21, state=_select({"w": selected}))

    assert _input_names(model) == ["in_0", "w"]
    assert any(
        np.array_equal(numpy_helper.to_array(tensor), constant)
        for tensor in model.graph.initializer
    )


def test_overridable_default_carries_tensor_metadata() -> None:
    w = _weight()
    metadata = {"pkg.example.path": "blocks.0.weight", "pkg.example.frozen": "false"}

    model = to_onnx(
        lambda x: x * w,
        inputs=_INPUT_SPEC,
        opset=21,
        state=_select({"w": w}, metadata=metadata),
    )

    props = {
        entry.key: entry.value for entry in _initializer(model, "w").metadata_props
    }
    assert props == metadata
    assert [item.name for item in _session(model).get_overridable_initializers()] == [
        "w"
    ]


def test_non_overridable_state_is_a_plain_initializer() -> None:
    w = _weight()
    metadata = {"pkg.example.path": "weight"}

    model = to_onnx(
        lambda x: x * w,
        inputs=_INPUT_SPEC,
        opset=21,
        state=_select({"w": w}, metadata=metadata, overridable=False),
    )

    assert _input_names(model) == ["in_0"]
    props = {
        entry.key: entry.value for entry in _initializer(model, "w").metadata_props
    }
    assert props == metadata
    assert not _session(model).get_overridable_initializers()
    np.testing.assert_allclose(_run(model, {"in_0": _DATA})[0], _DATA * np.asarray(w))


def test_selected_array_returned_directly_keeps_its_default() -> None:
    w = _weight()

    model = to_onnx(
        lambda x: (x + 1.0, w),
        inputs=_INPUT_SPEC,
        opset=21,
        state=_select({"w": w}),
    )
    onnx.checker.check_model(model)

    outputs = _run(model, {"in_0": _DATA})
    np.testing.assert_allclose(outputs[1], np.asarray(w))


def test_selected_array_nothing_reads_is_dropped() -> None:
    read, unread = _weight(), jnp.asarray([5.0, 7.0], dtype=jnp.float32)

    def fn(x: jax.Array) -> jax.Array:
        _ = unread + 1.0
        return x * read

    model = to_onnx(
        fn,
        inputs=_INPUT_SPEC,
        opset=21,
        state=_select({"read": read, "unread": unread}),
    )

    assert _input_names(model) == ["in_0", "read"]
    assert [tensor.name for tensor in model.graph.initializer] == ["read"]


def test_state_dtype_follows_the_captured_array() -> None:
    w = jnp.asarray([2, 3], dtype=jnp.int32)
    bf16 = jnp.asarray([0.5, 1.5], dtype=jnp.bfloat16)

    def fn(x: jax.Array) -> jax.Array:
        return x * w.astype(x.dtype) + bf16.astype(x.dtype)

    model = to_onnx(
        fn, inputs=_INPUT_SPEC, opset=21, state=_select({"w": w, "bf16": bf16})
    )

    assert _initializer(model, "w").data_type == onnx.TensorProto.INT32
    assert _initializer(model, "bf16").data_type == onnx.TensorProto.BFLOAT16
    types = {
        value.name: value.type.tensor_type.elem_type for value in model.graph.input
    }
    assert types["w"] == onnx.TensorProto.INT32
    assert types["bf16"] == onnx.TensorProto.BFLOAT16
    stored = numpy_helper.to_array(_initializer(model, "bf16"))
    assert stored.tobytes() == np.asarray(bf16).tobytes()


_FUNCTION_WEIGHT: jax.Array = jnp.asarray([2.0, 3.0], dtype=jnp.float32)


@onnx_function
def _scaled_inside_function(x: jax.Array) -> jax.Array:
    return x * _FUNCTION_WEIGHT


def test_selected_array_inside_onnx_function_is_rejected() -> None:
    with pytest.raises(ValueError, match="captured inside an ONNX Function"):
        to_onnx(
            lambda x: _scaled_inside_function(x),
            inputs=_INPUT_SPEC,
            opset=21,
            state=_select({"w": _FUNCTION_WEIGHT}),
        )


def test_unselected_array_inside_onnx_function_stays_a_constant() -> None:
    model = to_onnx(
        lambda x: _scaled_inside_function(x),
        inputs=_INPUT_SPEC,
        opset=21,
        state=lambda _: None,
    )

    assert _input_names(model) == ["in_0"]
    assert len(model.functions) == 1


def test_duplicate_state_name_is_rejected() -> None:
    first, second = _weight(), jnp.asarray([5.0, 7.0], dtype=jnp.float32)
    specs = {id(first): StateInput("w"), id(second): StateInput("w")}

    with pytest.raises(ValueError, match="already in use"):
        to_onnx(
            lambda x: x * first + second,
            inputs=_INPUT_SPEC,
            opset=21,
            state=lambda array: specs.get(id(array)),
        )


@pytest.mark.parametrize("name", ["in_0", "flag"])
def test_state_name_cannot_reuse_a_model_input(name: str) -> None:
    w = _weight()

    def fn(x: jax.Array, flag: bool = True) -> jax.Array:
        return x * w if flag else x

    with pytest.raises(ValueError, match="already in use"):
        to_onnx(
            fn,
            inputs=_INPUT_SPEC,
            input_params={"flag": True},
            opset=21,
            state=_select({name: w}),
        )


@pytest.mark.parametrize(
    "names",
    [{"input_names": ["w"]}, {"output_names": ["w"]}],
    ids=["input_names", "output_names"],
)
@pytest.mark.parametrize("overridable", [True, False])
def test_state_name_cannot_reuse_a_custom_io_name(
    names: dict[str, list[str]], overridable: bool
) -> None:
    w = _weight()

    with pytest.raises(ValueError, match="collide"):
        to_onnx(
            lambda x: x * w,
            inputs=_INPUT_SPEC,
            opset=21,
            state=_select({"w": w}, overridable=overridable),
            **names,
        )


def test_state_name_wins_over_a_generated_value_name() -> None:
    w, constant = _weight(), jnp.asarray([5.0, 7.0], dtype=jnp.float32)

    model = to_onnx(
        lambda x: x * w + constant,
        inputs=_INPUT_SPEC,
        opset=21,
        state=_select({"const_0": w}),
    )

    assert _input_names(model) == ["in_0", "const_0"]
    assert np.array_equal(numpy_helper.to_array(_initializer(model, "const_0")), w)
    np.testing.assert_allclose(
        _run(model, {"in_0": _DATA})[0], _DATA * np.asarray(w) + np.asarray(constant)
    )


def test_complex_state_is_rejected() -> None:
    w = jnp.asarray([1.0 + 2.0j], dtype=jnp.complex64)

    with pytest.raises(TypeError, match="cannot be complex"):
        to_onnx(
            lambda x: jnp.real(x * w),
            inputs=_INPUT_SPEC,
            opset=21,
            state=_select({"w": w}),
        )


def test_callback_must_return_state_input_or_none() -> None:
    w = _weight()

    with pytest.raises(TypeError, match="must return StateInput"):
        to_onnx(
            lambda x: x * w,
            inputs=_INPUT_SPEC,
            opset=21,
            state=lambda _: "w",  # type: ignore[arg-type, return-value]
        )


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"name": ""}, ValueError),
        ({"name": "w", "metadata": {"key": 1}}, TypeError),
    ],
    ids=["empty_name", "non_string_metadata"],
)
def test_state_input_validates_its_fields(
    kwargs: dict[str, Any], error: type[Exception]
) -> None:
    with pytest.raises(error):
        StateInput(**kwargs)


def test_callback_selecting_nothing_matches_a_plain_export() -> None:
    w = _weight()

    def fn(x: jax.Array) -> jax.Array:
        return jax.jit(lambda y: y * w)(x)

    plain = to_onnx(fn, inputs=_INPUT_SPEC, opset=21)
    selected_nothing = to_onnx(fn, inputs=_INPUT_SPEC, opset=21, state=lambda _: None)

    assert selected_nothing.SerializeToString() == plain.SerializeToString()


def test_ir_return_keeps_defaults_on_inputs() -> None:
    w = _weight()

    model = to_onnx(
        lambda x: x * w,
        inputs=_INPUT_SPEC,
        opset=21,
        state=_select({"w": w}, metadata={"key": "value"}),
        return_mode="ir",
    )

    assert isinstance(model, ir.Model)
    value = model.graph.initializers["w"]
    assert value.is_graph_input()
    assert value.const_value is not None
    assert dict(value.const_value.metadata_props) == {"key": "value"}


def test_file_export_keeps_metadata_with_external_data(tmp_path: Path) -> None:
    w = jnp.ones((400_000,), dtype=jnp.float32)
    path = tmp_path / "model.onnx"

    to_onnx(
        lambda x: x * w,
        inputs=[jax.ShapeDtypeStruct((400_000,), jnp.float32)],
        opset=21,
        state=_select({"w": w}, metadata={"key": "value"}),
        return_mode="file",
        output_path=path,
    )

    assert (tmp_path / "model.onnx.data").exists()
    assert allclose(lambda x: x * w, str(path), [(400_000,)])[0]
    reloaded = onnx.load(path)
    props = {
        entry.key: entry.value for entry in _initializer(reloaded, "w").metadata_props
    }
    assert props == {"key": "value"}
    assert _input_names(reloaded) == ["in_0", "w"]
