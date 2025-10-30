# Owner(s): ["oncall: pt2"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import ExitStack

import torch
import torch.nn as nn
from torch._decomp import decomposition_table
from torch._export import CompilerPipeline
from torch._functorch.partitioners import default_partition
from torch.testing._internal.common_utils import run_tests, TestCase


class TestCompilerPipeline(TestCase):
    def test_simple_linear_stage_by_stage(self):
        """Test calling CompilerPipeline methods individually."""

        class SimpleLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 2)

            def forward(self, x):
                return self.linear(x)

        model = SimpleLinear()
        inputs = (torch.randn(4, 3, requires_grad=True),)

        # Step 1: Create CompilerPipeline
        pipeline = CompilerPipeline(model, inputs)

        # Step 2: Capture graph for export
        gm = pipeline.capture_graph()

        with ExitStack() as stack:
            # Step 3: Generate joint graph with descriptors
            pipeline.generate_joint_graph(stack, decompositions=decomposition_table)

            def nop_compiler(gm, args):
                return gm.forward

            # Step 4: Partition the joint graph
            partition_output = pipeline.partition(
                default_partition,
                pipeline.aot_graph_capture.graph_module,
                pipeline.aot_graph_capture.updated_flat_args,
            )

            # Verify partition output
            self.assertIsNotNone(partition_output.fw_module)
            self.assertIsNotNone(partition_output.bw_module)
            self.assertGreater(partition_output.num_fw_outs_saved_for_bw, 0)
            self.assertIsInstance(partition_output.adjusted_flat_args, list)

            # Step 5: Compile forward
            fw_compile_output = pipeline.fw_compile(
                nop_compiler,
                partition_output.fw_module,
                partition_output.adjusted_flat_args,
                partition_output.num_fw_outs_saved_for_bw,
            )

            self.assertIsNotNone(fw_compile_output.compiled_fw_func)

            # Step 6: Compile backward
            bw_compile_output = pipeline.bw_compile(
                nop_compiler,
                partition_output.bw_module,
                fw_compile_output.fwd_output_strides,
                partition_output.num_symints_saved_for_bw,
            )

            self.assertIsNotNone(bw_compile_output.lazy_backward_info)

            # Step 7: Create the final autograd function (with clean calling convention)
            model_fn = pipeline.make_autograd_function(
                flat_args=pipeline.aot_state.flat_args,
                wrappers=pipeline.aot_graph_capture.wrappers,
                compiled_fw_func=fw_compile_output.compiled_fw_func,
                compiled_bw_func=bw_compile_output.compiled_bw_func,
                lazy_backward_info=bw_compile_output.lazy_backward_info,
                indices_of_inps_to_detach=partition_output.indices_of_inps_to_detach,
                num_symints_saved_for_bw=partition_output.num_symints_saved_for_bw,
            )
            self.assertIsNotNone(model_fn)

        # Test functional correctness: model_fn should produce same results as original model

        # Test forward
        expected_output = model(*inputs)
        actual_output = model_fn(*inputs)
        torch.testing.assert_close(actual_output, expected_output)

        # Test backward - check that gradients match
        # Create fresh inputs for both eager and compiled
        inputs_eager = (torch.randn(4, 3, requires_grad=True),)
        inputs_compiled = (inputs_eager[0].detach().clone().requires_grad_(True),)

        # Run eager backward
        out_eager = model(*inputs_eager)
        out_eager.sum().backward()

        # Run compiled backward
        out_compiled = model_fn(*inputs_compiled)
        out_compiled.sum().backward()

        # Compare gradients for input
        torch.testing.assert_close(inputs_eager[0].grad, inputs_compiled[0].grad)

        # Compare gradients for parameters (note: gm has the parameters)
        for (name_eager, param_eager), (name_compiled, param_compiled) in zip(
            model.named_parameters(), gm.named_parameters()
        ):
            self.assertEqual(name_eager, name_compiled)
            torch.testing.assert_close(param_eager.grad, param_compiled.grad)

    def test_simple_linear_with_structured_io(self):
        """Test calling CompilerPipeline with structured dict input and tuple output."""

        class SimpleLinearDict(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(3, 2)
                self.linear2 = nn.Linear(4, 2)

            def forward(self, inputs):
                # Take a dict with two tensors and return a tuple
                x = self.linear1(inputs["x"])
                y = self.linear2(inputs["y"])
                return (x + y, x - y, x * y)

        model = SimpleLinearDict()
        inputs = ({"x": torch.randn(4, 3, requires_grad=True), "y": torch.randn(4, 4, requires_grad=True)},)

        # Step 1: Create CompilerPipeline
        pipeline = CompilerPipeline(model, inputs)

        # Step 2: Capture graph for export
        gm = pipeline.capture_graph()

        with ExitStack() as stack:
            # Step 3: Generate joint graph with descriptors
            pipeline.generate_joint_graph(stack, decompositions=decomposition_table)

            def nop_compiler(gm, args):
                return gm.forward

            # Step 4: Partition the joint graph
            partition_output = pipeline.partition(
                default_partition,
                pipeline.aot_graph_capture.graph_module,
                pipeline.aot_graph_capture.updated_flat_args,
            )

            # Verify partition output
            self.assertIsNotNone(partition_output.fw_module)
            self.assertIsNotNone(partition_output.bw_module)
            self.assertGreater(partition_output.num_fw_outs_saved_for_bw, 0)
            self.assertIsInstance(partition_output.adjusted_flat_args, list)

            # print("===JOINT===")
            # aot_graph_capture.graph_module.print_readable()

            # print("===FORWARD===")
            # partition_output.fw_module.print_readable()
            # print("===BACKWARD===")
            # partition_output.bw_module.print_readable()

            # print("===PARAMS SPEC===")
            # print(joint_with_descriptors.params_spec)
            # print("===BUFFERS SPEC===")
            # print(joint_with_descriptors.buffers_spec)
            # print("===IN SPEC===")
            # print(joint_with_descriptors.in_spec)
            # print("===OUT SPEC===")
            # print(joint_with_descriptors.out_spec)

            # def num_inputs_outputs(my_graph_module):
            #     num_inputs = 0
            #     num_outputs = 0
            #     for node in my_graph_module.graph.nodes:
            #         if node.op == "placeholder":
            #             num_inputs += 1
            #         elif node.op == "output":
            #             num_outputs = len(node.args[0])

            #     return num_inputs, num_outputs

            # print("Summary (fw_module):", num_inputs_outputs(partition_output.fw_module))
            # print("Summary (bw_module):", num_inputs_outputs(partition_output.bw_module))
            # print("Summary (params_spec):", len(joint_with_descriptors.params_spec))
            # print("Summary (buffers_spec):", len(joint_with_descriptors.buffers_spec))
            # print("Summary (in_spec):", joint_with_descriptors.in_spec.num_leaves)
            # print("Summary (out_spec):", joint_with_descriptors.out_spec.num_leaves)

            # Step 5: Compile forward
            fw_compile_output = pipeline.fw_compile(
                nop_compiler,
                partition_output.fw_module,
                partition_output.adjusted_flat_args,
                partition_output.num_fw_outs_saved_for_bw,
            )

            self.assertIsNotNone(fw_compile_output.compiled_fw_func)

            # Step 6: Compile backward
            bw_compile_output = pipeline.bw_compile(
                nop_compiler,
                partition_output.bw_module,
                fw_compile_output.fwd_output_strides,
                partition_output.num_symints_saved_for_bw,
            )

            self.assertIsNotNone(bw_compile_output.lazy_backward_info)

            # Step 7: Create the final autograd function (with clean calling convention)
            model_fn = pipeline.make_autograd_function(
                flat_args=pipeline.aot_state.flat_args,
                wrappers=pipeline.aot_graph_capture.wrappers,
                compiled_fw_func=fw_compile_output.compiled_fw_func,
                compiled_bw_func=bw_compile_output.compiled_bw_func,
                lazy_backward_info=bw_compile_output.lazy_backward_info,
                indices_of_inps_to_detach=partition_output.indices_of_inps_to_detach,
                num_symints_saved_for_bw=partition_output.num_symints_saved_for_bw,
            )
            self.assertIsNotNone(model_fn)

        # Test functional correctness: model_fn should preserve dict calling convention and tuple output

        # Test forward with dict input and tuple output
        expected_output = model(*inputs)
        actual_output = model_fn(*inputs)

        # Verify we got a tuple with 2 elements
        self.assertIsInstance(expected_output, tuple)
        self.assertIsInstance(actual_output, tuple)
        self.assertEqual(len(expected_output), 3)
        self.assertEqual(len(actual_output), 3)

        # Verify each element of the tuple matches
        for actual, expected in zip(actual_output, expected_output):
            torch.testing.assert_close(actual, expected)

        # Test backward - check that gradients match
        # Create fresh inputs for both eager and compiled
        inputs_eager = (
            {
                "x": torch.randn(4, 3, requires_grad=True),
                "y": torch.randn(4, 4, requires_grad=True),
            },
        )
        inputs_compiled = (
            {
                "x": inputs_eager[0]["x"].detach().clone().requires_grad_(True),
                "y": inputs_eager[0]["y"].detach().clone().requires_grad_(True),
            },
        )

        # Run eager backward (sum over both tuple outputs)
        out_eager = model(*inputs_eager)
        sum(x.sum() for x in out_eager).backward()

        # Run compiled backward (sum over both tuple outputs)
        out_compiled = model_fn(*inputs_compiled)
        sum(x.sum() for x in out_compiled).backward()

        # Compare gradients for inputs
        for input_eager, input_compiled in zip(
            inputs_eager[0].values(), inputs_compiled[0].values()
        ):
            torch.testing.assert_close(input_eager.grad, input_compiled.grad)

        # Compare gradients for parameters (note: gm has the parameters)
        for (name_eager, param_eager), (name_compiled, param_compiled) in zip(
            model.named_parameters(), gm.named_parameters()
        ):
            self.assertEqual(name_eager, name_compiled)
            torch.testing.assert_close(param_eager.grad, param_compiled.grad)

    def test_conv_bn_stage_by_stage(self):
        """Test CompilerPipeline stage-by-stage with conv+batchnorm model."""

        class ConvBN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 3, 3, padding=1)
                self.bn = nn.BatchNorm2d(3)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return torch.relu(x)

        model = ConvBN()
        model.eval()  # Use eval mode to avoid buffer mutations in functional correctness test
        inputs = (torch.randn(2, 1, 4, 4, requires_grad=True),)

        # Step 1: Create CompilerPipeline
        pipeline = CompilerPipeline(model, inputs)

        # Step 2: Capture graph for export
        gm = pipeline.capture_graph()

        with ExitStack() as stack:
            # Step 3: Generate joint graph with descriptors
            pipeline.generate_joint_graph(stack, decompositions=decomposition_table)

            def nop_compiler(gm, args):
                return gm.forward

            # Step 4: Partition
            partition_output = pipeline.partition(
                default_partition,
                pipeline.aot_graph_capture.graph_module,
                pipeline.aot_graph_capture.updated_flat_args,
            )

            self.assertIsNotNone(partition_output.fw_module)
            self.assertIsNotNone(partition_output.bw_module)

            # Step 5: Compile forward and backward
            fw_compile_output = pipeline.fw_compile(
                nop_compiler,
                partition_output.fw_module,
                partition_output.adjusted_flat_args,
                partition_output.num_fw_outs_saved_for_bw,
            )

            bw_compile_output = pipeline.bw_compile(
                nop_compiler,
                partition_output.bw_module,
                fw_compile_output.fwd_output_strides,
                partition_output.num_symints_saved_for_bw,
            )

            self.assertIsNotNone(fw_compile_output.compiled_fw_func)
            self.assertIsNotNone(bw_compile_output.lazy_backward_info)

            # Step 6: Create the final autograd function (with clean calling convention)
            model_fn = pipeline.make_autograd_function(
                flat_args=pipeline.aot_state.flat_args,
                wrappers=pipeline.aot_graph_capture.wrappers,
                compiled_fw_func=fw_compile_output.compiled_fw_func,
                compiled_bw_func=bw_compile_output.compiled_bw_func,
                lazy_backward_info=bw_compile_output.lazy_backward_info,
                indices_of_inps_to_detach=partition_output.indices_of_inps_to_detach,
                num_symints_saved_for_bw=partition_output.num_symints_saved_for_bw,
            )
            self.assertIsNotNone(model_fn)

        # Test forward
        expected_output = model(*inputs)
        actual_output = model_fn(*inputs)
        torch.testing.assert_close(actual_output, expected_output)

        # Test backward - check that gradients match
        # Create fresh inputs for both eager and compiled
        inputs_eager = (torch.randn(2, 1, 4, 4, requires_grad=True),)
        inputs_compiled = (inputs_eager[0].detach().clone().requires_grad_(True),)

        # Run eager backward
        out_eager = model(*inputs_eager)
        out_eager.sum().backward()

        # Run compiled backward
        out_compiled = model_fn(*inputs_compiled)
        out_compiled.sum().backward()

        # Compare gradients for input
        torch.testing.assert_close(inputs_eager[0].grad, inputs_compiled[0].grad)

        # Compare gradients for parameters (note: gm has the parameters)
        for (name_eager, param_eager), (name_compiled, param_compiled) in zip(
            model.named_parameters(), gm.named_parameters()
        ):
            self.assertEqual(name_eager, name_compiled)
            torch.testing.assert_close(param_eager.grad, param_compiled.grad)

    def test_inference_with_structured_io(self):
        """Test inference compilation path with structured dict input and tuple output."""

        class SimpleLinearDict(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(3, 2)
                self.linear2 = nn.Linear(4, 2)

            def forward(self, inputs):
                # Take a dict with two tensors and return a tuple
                x = self.linear1(inputs["x"])
                y = self.linear2(inputs["y"])
                return (x + y, x - y, x * y)

        model = SimpleLinearDict()
        model.eval()  # Set to eval mode for inference
        # Inputs do NOT require grad for inference path
        inputs = ({
            "x": torch.randn(4, 3),
            "y": torch.randn(4, 4)
        },)

        # Step 1: Create CompilerPipeline
        pipeline = CompilerPipeline(model, inputs)

        # Step 2: Capture graph for export
        gm = pipeline.capture_graph()

        # Params and buffers do NOT require grad for inference path
        with torch.no_grad(), ExitStack() as stack:
            # Step 3: Generate joint graph with descriptors
            pipeline.generate_joint_graph(stack, decompositions=decomposition_table)

            def nop_compiler(gm, args):
                return gm.forward

            # pipeline.aot_graph_capture.graph_module.print_readable()

            # Step 4: Compile for inference (no partitioning needed)
            compiled_fw = pipeline.inference_compile(
                nop_compiler,
                pipeline.aot_graph_capture.graph_module,
                pipeline.aot_graph_capture.updated_flat_args,
            )

            self.assertIsNotNone(compiled_fw)

            # Step 6: Create the final inference function (with clean calling convention)
            model_fn = pipeline.make_inference_function(
                compiled_fw=compiled_fw,
                wrappers=pipeline.aot_graph_capture.wrappers,
                entry=None,
            )
            self.assertIsNotNone(model_fn)

        # Test functional correctness: model_fn should preserve dict calling convention and tuple output

        # Test forward with dict input and tuple output
        expected_output = model(*inputs)
        actual_output = model_fn(*inputs)

        # Verify we got a tuple with 3 elements
        self.assertIsInstance(expected_output, tuple)
        self.assertIsInstance(actual_output, tuple)
        self.assertEqual(len(expected_output), 3)
        self.assertEqual(len(actual_output), 3)

        # Verify each element of the tuple matches
        for actual, expected in zip(actual_output, expected_output):
            torch.testing.assert_close(actual, expected)

    def test_with_identity_passes(self):
        """Test applying identity passes between compilation stages."""

        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 2)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModule()
        inputs = (torch.randn(4, 3, requires_grad=True),)

        # Identity pass that returns the graph unchanged
        def identity_pass(gm):
            """An identity pass that marks the graph as processed."""
            gm._identity_pass_applied = True
            return gm

        # Create CompilerPipeline
        pipeline = CompilerPipeline(model, inputs)

        # Capture and export
        gm = pipeline.capture_graph()

        with ExitStack() as stack:
            pipeline.generate_joint_graph(stack, decompositions=decomposition_table)

            # Apply identity pass to joint graph
            pipeline.aot_graph_capture.graph_module = identity_pass(pipeline.aot_graph_capture.graph_module)
            self.assertTrue(hasattr(pipeline.aot_graph_capture.graph_module, '_identity_pass_applied'))

            def nop_compiler(gm, args):
                return gm.forward

            # Partition
            partition_output = pipeline.partition(
                default_partition,
                pipeline.aot_graph_capture.graph_module,
                pipeline.aot_graph_capture.updated_flat_args,
            )

            # Apply identity pass to forward and backward modules
            partition_output.fw_module = identity_pass(partition_output.fw_module)
            partition_output.bw_module = identity_pass(partition_output.bw_module)

            self.assertTrue(hasattr(partition_output.fw_module, '_identity_pass_applied'))
            self.assertTrue(hasattr(partition_output.bw_module, '_identity_pass_applied'))

            # Continue compilation
            fw_compile_output = pipeline.fw_compile(
                nop_compiler,
                partition_output.fw_module,
                partition_output.adjusted_flat_args,
                partition_output.num_fw_outs_saved_for_bw,
            )

            bw_compile_output = pipeline.bw_compile(
                nop_compiler,
                partition_output.bw_module,
                fw_compile_output.fwd_output_strides,
                partition_output.num_symints_saved_for_bw,
            )

            self.assertIsNotNone(fw_compile_output.compiled_fw_func)
            self.assertIsNotNone(bw_compile_output.lazy_backward_info)

            # Create the final autograd function (with clean calling convention)
            model_fn = pipeline.make_autograd_function(
                flat_args=pipeline.aot_state.flat_args,
                wrappers=pipeline.aot_graph_capture.wrappers,
                compiled_fw_func=fw_compile_output.compiled_fw_func,
                compiled_bw_func=bw_compile_output.compiled_bw_func,
                lazy_backward_info=bw_compile_output.lazy_backward_info,
                indices_of_inps_to_detach=partition_output.indices_of_inps_to_detach,
                num_symints_saved_for_bw=partition_output.num_symints_saved_for_bw,
            )
            self.assertIsNotNone(model_fn)

        # Test functional correctness: model_fn should produce same results as original model
        expected_output = model(*inputs)
        actual_output = model_fn(*inputs)
        torch.testing.assert_close(actual_output, expected_output)


if __name__ == "__main__":
    run_tests()
