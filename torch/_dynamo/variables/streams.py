import collections
from typing import Any, Optional

import torch
from torch.fx import Proxy

from .. import graph_break_hints
from ..bytecode_transformation import create_call_function
from ..exc import TYPE_CHECKING, unimplemented_v2
from ..source import AttrSource, CallFunctionNoArgsSource, TorchSource
from .base import VariableTracker
from .constant import ConstantVariable
from .ctx_manager import FxTracebackAnnotateVariable
from .lazy import LazyVariableTracker


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator

    from ..codegen import PyCodegen

from torch._library.custom_ops import custom_op


Tensor = torch.Tensor


@custom_op("streams::fork", mutates_args=())
def fork_stream(
    from_index: int,
    from_device: torch.device,
    to_index: int,
    to_device: torch.device,
) -> None:
    pass


@fork_stream.register_fake
def _(
    from_index: int,
    from_device: torch.device,
    to_index: int,
    to_device: torch.device,
) -> None:
    pass


@custom_op("streams::join", mutates_args=())
def join_stream(
    from_index: int,
    from_device: torch.device,
    to_index: int,
    to_device: torch.device,
) -> None:
    pass


@join_stream.register_fake
def _(
    from_index: int,
    from_device: torch.device,
    to_index: int,
    to_device: torch.device,
) -> None:
    pass


class SymbolicStreamState:
    """Track the currently entered stream if any"""

    def __init__(self) -> None:
        from ..source import CurrentStreamSource

        cur_stack: list[StreamVariable] = []
        if torch.accelerator.is_available():
            stream_var = LazyVariableTracker.create(
                torch.accelerator.current_stream(),
                source=CurrentStreamSource(torch.accelerator.current_stream().device),
            )
            cur_stack = [stream_var]  # type: ignore[list-item]

        self.cur_stream_stack: collections.deque[StreamVariable] = collections.deque(
            cur_stack
        )

    def enter_stream(self, stream: "StreamVariable") -> None:
        self.cur_stream_stack.append(stream)

    def exit_stream(self) -> None:
        self.cur_stream_stack.pop()

    def cur_stream(self, device: Optional[torch.device] = None) -> "StreamVariable":
        if device is not None:
            for stream in reversed(self.cur_stream_stack):
                if stream.device == device:
                    return stream

        return self.cur_stream_stack[-1]

    def in_stream_context(self) -> bool:
        return len(self.cur_stream_stack) > 0


class StreamContextVariable(FxTracebackAnnotateVariable):
    """This represents torch.cuda.StreamContext"""

    @staticmethod
    def create(
        tx: "InstructionTranslator",
        stream_to_enter: "StreamVariable",
        **kwargs: dict[str, Any],
    ) -> "StreamContextVariable":
        return StreamContextVariable(
            stream_to_enter,
            **kwargs,
        )

    def __init__(
        self,
        stream: Optional["StreamVariable"],
        **kwargs: dict[str, Any],
    ) -> None:
        self.stream = stream
        super().__init__(
            target_values={"stream": self.get_stream().user_object_index},
            initial_values=None,
            **kwargs,
        )

    def enter(
        self, tx: "InstructionTranslator", *args: tuple[Any]
    ) -> "VariableTracker":
        # to stream, from stream is the order of the arguments
        # we are entering the target, and leaving the initial stream
        tx.symbolic_stream_state.enter_stream(self.get_stream())
        return super().enter(tx)

    def exit(self, tx: "InstructionTranslator", *args: tuple[Any]) -> "VariableTracker":
        # to stream, from stream is the order of the arguments
        # we are leaving the target, and entering the initial stream
        tx.symbolic_stream_state.exit_stream()
        return super().exit(tx, *args)

    def supports_graph_breaks(self) -> bool:
        return True

    def get_stream(self) -> "StreamVariable":
        assert self.stream, "Stream context should have a separate stream"
        return self.stream


class StreamVariable(StreamContextVariable):
    """Represents the device-agnostic torch.Stream class"""

    def __init__(
        self,
        proxy: Proxy,
        value: torch.Stream,
        **kwargs: Any,
    ) -> None:
        # Index into the user object table
        # used to pass arbitrary objects to the graph
        user_object_index = kwargs.pop("user_obj_index", None)
        if proxy is not None and "example_value" in proxy.node.meta:
            assert proxy.node.meta["example_value"] == value

        self.proxy = proxy
        self.value = value
        # pyrefly: ignore [read-only]
        self.device = value.device
        # pyrefly: ignore [read-only]
        self.user_object_index = user_object_index
        super().__init__(None, **kwargs)

    def python_type(self) -> type:
        return torch.Stream

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> "VariableTracker":
        assert hasattr(self.value, name), f"no stream method found named {name}"

        from ..utils import cmp_name_to_op_mapping, proxy_args_kwargs
        from .builder import wrap_fx_proxy_cls

        if name in ("wait_stream", "synchronize", "wait_event"):
            tx.output.create_proxy(
                "call_method", name, *proxy_args_kwargs([self] + args, kwargs)
            )
            return ConstantVariable(None)
        elif name == "query":
            return wrap_fx_proxy_cls(
                target_cls=ConstantVariable,
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_method", name, *proxy_args_kwargs([self] + args, kwargs)
                ),
            )
        elif name == "record_event":
            return wrap_fx_proxy_cls(
                target_cls=EventVariable,
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_method", name, *proxy_args_kwargs([self] + args, kwargs)
                ),
            )
        elif name in cmp_name_to_op_mapping and len(args) == 1 and not kwargs:
            from ..guards import GuardBuilder, install_guard

            if self.source:
                install_guard(self.source.make_guard(GuardBuilder.EQUALS_MATCH))

            # NB : Checking for mutation is necessary because we compare
            # constant values
            other = args[0]
            if not isinstance(other, StreamVariable):
                return ConstantVariable.create(NotImplemented)

            if other.source:
                install_guard(self.source.make_guard(GuardBuilder.EQUALS_MATCH))
            return ConstantVariable.create(
                cmp_name_to_op_mapping[name](self.value, other.value)  # type: ignore[arg-type]
            )

        return super().call_method(tx, name, args, kwargs)

    def as_proxy(self) -> Proxy:
        return self.proxy

    def module_name(self) -> str:
        return "torch._C"

    def fn_name(self) -> str:
        return "Stream"

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # If we got here, this stream is fully subsumed by the graph - this means it is
        # not an input or global
        assert not self.source
        if self.user_object_index is not None:
            codegen.add_push_null(
                lambda: codegen.load_import_from(
                    torch._dynamo.graph_bytecode_inputs.__name__,
                    "get_external_object_by_index",
                )
            )
            codegen.append_output(codegen.create_load_const(self.user_object_index))
            codegen.extend_output(create_call_function(1, False))
        else:
            # TODO mlazos: evaluate if we still need this
            prefix = f"_stream_{self.device}"
            name = codegen.tx.output.install_global_by_id(prefix, self.value)
            codegen.append_output(codegen.create_load_global(name, add=True))

    def get_stream(self) -> "StreamVariable":
        return self

    @staticmethod
    def construct_in_graph_stream(index: int, codegen: "PyCodegen") -> None:
        # Use source to create the right bytecode, this
        # isn't an actual input
        source = CallFunctionNoArgsSource(AttrSource(TorchSource(), "Stream"))
        codegen(source)


class EventVariable(VariableTracker):
    def __init__(self, proxy: Proxy, value: torch.Event, **kwargs: Any) -> None:
        if proxy is not None and "example_value" in proxy.node.meta:
            assert proxy.node.meta["example_value"] == value
        super().__init__(**kwargs)
        self.proxy = proxy
        self.value = value

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from ..utils import proxy_args_kwargs
        from .builder import wrap_fx_proxy_cls

        if name in ("wait", "record", "synchronize"):
            tx.output.create_proxy(
                "call_method", name, *proxy_args_kwargs([self] + args, kwargs)
            )
            return ConstantVariable(None)
        elif name == "query":
            return wrap_fx_proxy_cls(
                target_cls=ConstantVariable,
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_method", name, *proxy_args_kwargs([self] + args, kwargs)
                ),
            )
        else:
            method_name = (
                f"{type(self.value).__module__}.{type(self.value).__qualname__}.{name}"
            )
            unimplemented_v2(
                gb_type="Unsupported event method",
                context=str(name),
                explanation=f"Dynamo doesn't support tracing the {method_name} method. "
                f"We currently support wait, record, synchronize, and query.",
                hints=[
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

    def as_proxy(self) -> Proxy:
        return self.proxy

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # If we got here, this event is fully subsumed by the graph - this means it is
        # not an input or global
        assert not self.source
        # Similar to stream handling, we lift the event into a global and then codegen bytecode to load it from there.
        prefix = "_event"
        name = codegen.tx.output.install_global_by_id(prefix, self.value)
        codegen.append_output(codegen.create_load_global(name, add=True))
