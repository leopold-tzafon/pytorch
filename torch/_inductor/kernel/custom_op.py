# Owner(s): ["module: inductor"]

import functools
import logging
from typing import Any, Callable, Optional, Union

import torch
from torch import _ops
from torch._inductor.codegen.subgraph import SubgraphTemplate
from torch._inductor.ir import Buffer, FixedLayout, ir_node_to_tensor, TensorBox
from torch._inductor.lowering import lowerings, validate_ir
from torch._inductor.select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
)
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)


class CustomOpConfig:
    """Config for custom op autotuning.

    Specifies optional decomposition function with parameter values.
    Each config creates exactly one variant.

    Args:
        decomposition: Optional functions to autotune. If not provided, default will be used.
        **params: Parameters passed to the function

    Examples:
        CustomOpConfig(attention_impl, head_dim=32, method='chunked')
        CustomOpConfig(head_dim=32, method='chunked')
    """

    def __init__(
        self,
        decomposition: Optional[Callable[..., Any]] = None,
        **params: Any,
    ):
        if decomposition is not None and not callable(decomposition):
            raise TypeError(
                f"decomposition must be callable, got {type(decomposition)}"
            )

        self.decomposition = decomposition
        self.params = params

    def get_decomposition(
        self, default_impl: Optional[Callable[..., Any]] = None
    ) -> Callable[..., Any]:
        """Return the decomposition function for this config.
        When decomposition is not specified, return the default implementation
        from the custom op's registration.
        """
        if self.decomposition is not None:
            return self.decomposition

        # If no decomposition specified, get Python implementation from custom op
        if default_impl and isinstance(default_impl, (_ops.OpOverload, str)):
            from torch._library.custom_ops import _maybe_get_opdef

            op_def = _maybe_get_opdef(default_impl)
            if op_def is not None and hasattr(op_def, "_init_fn"):
                return op_def._init_fn

        raise TypeError(
            f"Could not extract Python implementation from {default_impl}. "
            f"Please register customop or provide a decomposition function."
        )

    def get_name(self) -> str:
        """Get the name for this config variant."""
        param_suffix = (
            "_".join(f"{k}_{v}" for k, v in sorted(self.params.items()))
            if self.params
            else ""
        )

        base_name = self.decomposition.__name__ if self.decomposition else "default"

        return f"{base_name}_{param_suffix}" if param_suffix else base_name

    def __repr__(self) -> str:
        if self.params:
            params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
            decomp_name = (
                self.decomposition.__name__ if self.decomposition else "default"
            )
            return f"CustomOpConfig({decomp_name}, {params_str})"
        decomp_name = self.decomposition.__name__ if self.decomposition else "default"
        return f"CustomOpConfig({decomp_name})"


__all__ = [
    "autotune_custom_op",
    "register_custom_op_autotuning",
    "CustomOpConfig",
]


def _extract_tensor_inputs(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    """Extract tensor inputs from mixed args/kwargs.
    Separates tensors (for autotuning input_nodes) from non-tensor parameters.
    Non-tensor kwargs are later functools.partial'd into decomposition functions.

    Args:
        args: Positional arguments (mix of tensors and scalars)
        kwargs: Keyword arguments (mix of tensors and scalars)

    Returns:
        Tuple of (tensor_inputs_list, non_tensor_kwargs)
    """
    tensor_inputs = []
    non_tensor_kwargs = {}

    # Process args and kwargs: separate tensor inputs and non tensor args
    for i, arg in enumerate(args):
        if isinstance(arg, (TensorBox, Buffer)):
            tensor_inputs.append(arg)
        else:
            # Add non-tensor positional args to kwargs with generated names
            non_tensor_kwargs[f"arg_{i}"] = arg

    for key, value in kwargs.items():
        if isinstance(value, (TensorBox, Buffer)):
            tensor_inputs.append(value)
        else:
            non_tensor_kwargs[key] = value

    return tensor_inputs, non_tensor_kwargs


def _merge_config_and_runtime_kwargs(
    config_params: dict[str, Any],
    runtime_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Merge config parameters with runtime kwargs. Runtime kwargs take precedence.
       If there are conflicts, log a warning and use runtime value.

    Args:
        config_params: Parameters from CustomOpConfig
        runtime_kwargs: Runtime non-tensor kwargs from _extract_tensor_inputs

    Returns:
        Merged kwargs dictionary with runtime values taking precedence
    """
    merged_kwargs = config_params.copy()

    # Check for conflicts and let runtime kwargs dominate
    conflicts = OrderedSet(config_params.keys()).intersection(runtime_kwargs.keys())

    for key in conflicts:
        log.warning(
            "Parameter '%s' specified both in CustomOpConfig (%s) "
            "and at runtime (%s). Using runtime value.",
            key,
            config_params[key],
            runtime_kwargs[key],
        )

    # Runtime kwargs override config params
    merged_kwargs.update(runtime_kwargs)

    return merged_kwargs


def _create_user_input_gen_fns(
    inputs: list[Any],
    arg_names: list[str],
    user_input_gen_fns: dict[str, Callable[[torch.Tensor], torch.Tensor]],
) -> dict[int, Callable[[Any], torch.Tensor]]:
    """Convert user input generators from name-based to index-based format.
       Inductor autotune's input_gen_fns expects index of arg_names as key.

    Uses V.graph.sizevars.size_hints() to guess best for dynamic shapes.
    """
    from torch._inductor import config

    name_to_index = {name: i for i, name in enumerate(arg_names)}
    index_based_fns = {}

    for name, gen_fn in user_input_gen_fns.items():
        if name in name_to_index:
            index_based_fns[name_to_index[name]] = gen_fn
        else:
            log.warning(
                "Unknown argument name '%s' in input_gen_fns. "
                "Available argument names: %s",
                name,
                list(name_to_index.keys()),
            )

    def create_internal_input_gen_fn(
        user_function: Callable[[torch.Tensor], torch.Tensor], arg_name: str
    ) -> Callable[[Any], torch.Tensor]:
        """Create internal input generator that converts IR buffer to user's fake tensor."""

        def internal_input_gen_fn(ir_buffer: Any) -> torch.Tensor:
            raw_shape = ir_buffer.get_size()
            concrete_shape = V.graph.sizevars.size_hints(
                raw_shape, fallback=config.unbacked_symint_fallback
            )

            fake_tensor = torch.empty(
                concrete_shape, dtype=ir_buffer.get_dtype(), device="meta"
            )
            return user_function(fake_tensor)

        return internal_input_gen_fn

    return {
        i: create_internal_input_gen_fn(
            user_gen_fn, arg_names[i] if i < len(arg_names) else f"arg_{i}"
        )
        for i, user_gen_fn in index_based_fns.items()
        if i < len(inputs)
    }


def _create_fallback_choice(
    name: str,
    default_impl: Callable[..., Any],
    fake_output: torch.Tensor,
    kwargs: dict[str, Any],
) -> ExternKernelChoice:
    """Create fallback choice for default implementation."""

    def fallback_wrapper(*args: Any) -> Any:
        return default_impl(*args, **kwargs)

    return ExternKernelChoice(
        kernel=fallback_wrapper,
        name=f"{name}_fallback_default",
        has_out_variant=False,
        op_overload=default_impl,
        use_fallback_kernel=True,
    )


def autotune_custom_op(
    name: str,
    decompositions: list[Callable[..., Any]],
    inputs: list[Any],
    non_tensor_args: list[dict[str, Any]],
    default_impl: Optional[Callable[..., Any]] = None,
    user_input_gen_fns: Optional[
        dict[str, Callable[[torch.Tensor], torch.Tensor]]
    ] = None,
) -> Union[TensorBox, Any]:
    """Autotune custom operations by comparing multiple decomposition implementations.

    Currently supports SINGLE OUTPUT custom ops only.
    TODO: Add support for multiple output custom ops (tuple/list returns).

    This function generates multiple implementation choices for a custom operation and
    uses Inductor's autotuning system to select the best performing variant at runtime.

    Args:
        name: Unique identifier for the autotuning operation
        decompositions: List of alternative implementation functions to benchmark
        inputs: Input tensor IR nodes from compilation (TensorBox/Buffer objects)
        non_tensor_args: List of kwargs dicts, paired with corresponding decompositions arg
        default_impl: Original custom op implementation used as fallback
        user_input_gen_fns: Optional custom input generators for benchmarking.
                           Maps input indices to functions that take fake tensors
                           and return real tensors for performance measurement.

    Returns:
        IR node representing the optimized operation result

    Raises:
        TypeError: If decompositions is not a list/tuple
        RuntimeError: If no inputs or no valid choices generated
    """
    if not isinstance(decompositions, (list, tuple)):
        raise TypeError(
            f"decompositions must be a list or tuple of callables, got {type(decompositions)}"
        )

    if not inputs:
        raise RuntimeError(f"Custom op '{name}' requires tensor inputs for autotuning")

    if len(decompositions) != len(non_tensor_args):
        raise ValueError(
            f"decompositions and non_tensor_args must have same length, "
            f"got {len(decompositions)} decompositions and {len(non_tensor_args)} kwargs"
        )

    template = SubgraphTemplate(name=name)
    choices = template.generate_custom_op_choices(
        name=name,
        decompositions=decompositions,
        input_nodes=list(inputs),
        non_tensor_args=non_tensor_args,
    )

    # Add default implementation as fallback
    if default_impl and hasattr(default_impl, "_op"):
        fallback_name = f"{name}_fallback_default"
        from torch._inductor.select_algorithm import extern_kernels

        # Skip if extern_kernel already registered to avoid duplicate registration error
        if not hasattr(extern_kernels, fallback_name):
            with V.fake_mode:
                fake_inputs = [ir_node_to_tensor(inp) for inp in inputs]
                fallback_kwargs = non_tensor_args[0] if non_tensor_args else {}
                fake_output = default_impl(*fake_inputs, **fallback_kwargs)

            fallback_choice = _create_fallback_choice(
                name, default_impl, fake_output, fallback_kwargs
            )
            fallback_choice.maybe_append_choice(
                choices=choices,
                input_nodes=list(inputs),
                layout=FixedLayout(
                    device=fake_output.device,
                    dtype=fake_output.dtype,
                    size=fake_output.shape,
                    stride=fake_output.stride(),
                ),
            )

    if not choices:
        raise RuntimeError(f"No valid choices generated for {name}")

    # Convert user input generation functions to internal format
    input_gen_fns = {}
    if user_input_gen_fns:
        import inspect

        arg_names = (
            list(inspect.signature(decompositions[0]).parameters.keys())
            if decompositions
            else []
        )
        input_gen_fns = _create_user_input_gen_fns(
            inputs, arg_names, user_input_gen_fns
        )

    return autotune_select_algorithm(
        name=name,
        choices=choices,
        input_nodes=list(inputs),
        layout=choices[0].layout,
        input_gen_fns=input_gen_fns,
    )


def register_custom_op_autotuning(
    custom_op: torch._ops.OpOverload,
    configs: Union[list[CustomOpConfig], list[Callable[..., Any]]],
    name: Optional[str] = None,
    input_gen_fns: Optional[dict[str, Callable[[torch.Tensor], torch.Tensor]]] = None,
) -> None:
    """Register custom op for autotuning with custom_op configs where each config
    specifies a decomposition implementation function with its parameter values.

    Args:
        custom_op: Custom operation to register
        configs: List of CustomOpConfig objects or callable functions
        name: Operation name (default: "{op_name}_autotuned")
        input_gen_fns: Custom input generators for benchmarking

    Examples:
        register_custom_op_autotuning(
            torch.ops.mylib.attention.default,
            configs=[
                CustomOpConfig(attention_impl, head_dim=32, method='chunked'),
                CustomOpConfig(attention_impl, head_dim=64, method='tiled'),
                CustomOpConfig(fallback_impl),  # No params
            ],
            input_gen_fns={
                "query": lambda fake: torch.randn_like(fake, device='cuda'),
                "key": lambda fake: torch.randn_like(fake, device='cuda'),
                "value": lambda fake: torch.randn_like(fake, device='cuda'),
            }
        )
    """
    if not isinstance(configs, (list, tuple)):
        raise TypeError(f"configs must be a list or tuple, got {type(configs)}")

    processed_configs = []
    for config in configs:
        if isinstance(config, CustomOpConfig):
            processed_configs.append(config)
        else:
            raise TypeError(
                f"Each config must be a CustomOpConfig object, got {type(config)}"
            )

    if not processed_configs:
        raise ValueError("At least one config must be provided")

    if name is None:
        name = f"{custom_op._name}_autotuned"

    @functools.wraps(custom_op)
    def autotuning_lowering(*args: Any, **kwargs: Any) -> Any:
        """Inductor lowering function that replaces custom op calls with autotuned versions."""
        # Extract tensor inputs and non-tensor parameters (runtime kwargs)
        tensor_inputs, runtime_kwargs = _extract_tensor_inputs(args, kwargs)

        # Prepare decompositions and kwargs by merging customop config params with runtime kwargs
        decompositions = []
        non_tensor_args = []

        for config in processed_configs:
            decomp = config.get_decomposition(default_impl=custom_op)
            decompositions.append(decomp)

            # Merge config params with runtime kwargs (runtime takes precedence)
            merged_kwargs = _merge_config_and_runtime_kwargs(
                config.params, runtime_kwargs
            )
            non_tensor_args.append(merged_kwargs)

        result = autotune_custom_op(
            name=name,
            decompositions=decompositions,
            inputs=tensor_inputs,
            non_tensor_args=non_tensor_args,
            default_impl=custom_op,
            user_input_gen_fns=input_gen_fns,
        )

        validate_ir(result)
        return result

    lowerings[custom_op] = autotuning_lowering
