#!/usr/bin/env python
#=============================================================================
#
#  Copyright (c) 2015-2018 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
from converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from .fullyconnected import FullyConnectedLayerResolver
from converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence,
    NonConsumableConverterSequenceNode
)
from functools import reduce

class ReshapeLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, reshape_op):
            super(ReshapeLayerResolver.Descriptor, self).__init__('Reshape', name, nodes)
            self.reshape_op = reshape_op

    def __init__(self):
        sequence_reshape = GraphSequence([ConverterSequenceNode('root', ['Reshape', 'Squeeze', 'ExpandDims'])])
        sequence_reshape.set_outputs(['root'])

        sequence_flatten = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            ConverterSequenceNode('shape', ['Shape']),
            ConverterSequenceNode('slice_1', ['Slice']),
            ConverterSequenceNode('const_1', ['Const']),
            ConverterSequenceNode('const_2', ['Const']),
            ConverterSequenceNode('slice_2', ['Slice']),
            ConverterSequenceNode('const_3', ['Const']),
            ConverterSequenceNode('const_4', ['Const']),
            ConverterSequenceNode('prod', ['Prod']),
            ConverterSequenceNode('const_5', ['Const']),
            ConverterSequenceNode('expand_dims', ['ExpandDims']),
            ConverterSequenceNode('const_6', ['Const']),
            ConverterSequenceNode('concat', ['ConcatV2']),
            ConverterSequenceNode('const_7', ['Const']),
            ConverterSequenceNode('root', ['Reshape']),
        ])
        sequence_flatten.set_inputs('shape', ['input'])
        sequence_flatten.set_inputs('slice_1', ['shape', 'const_1', 'const_2'])
        sequence_flatten.set_inputs('slice_2', ['shape', 'const_3', 'const_4'])
        sequence_flatten.set_inputs('prod', ['slice_2', 'const_5'])
        sequence_flatten.set_inputs('expand_dims', ['prod', 'const_6'])
        sequence_flatten.set_inputs('concat', ['slice_1', 'expand_dims', 'const_7'])
        sequence_flatten.set_inputs('root', ['input', 'concat'])
        sequence_flatten.set_outputs(['root'])

        self.sequences = [sequence_reshape, sequence_flatten]

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for sequence in self.sequences:
            matches = graph_matcher.match_sequence(sequence)
            for match in matches:
                reshape_op = match['root']
                consumed_nodes = match.consumed_nodes
                descriptors.append(
                    ReshapeLayerResolver.Descriptor(str(reshape_op.name), consumed_nodes,
                                                    reshape_op))
        return descriptors


class ReshapeLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ReshapeLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors[:1])
        output_shape = converter_context.graph_helper.get_op_output_shape(descriptor.reshape_op)
        output_shape = output_shape[-4:] if len(output_shape) > 4 else output_shape
        return converter_context.model.add_reshape_layer(descriptor.output_names[0],
                                                         output_shape,
                                                         input_name,
                                                         descriptor.output_names[0])

    def transform_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        fc_outputs = [d for d in output_descriptors if isinstance(d, FullyConnectedLayerResolver.Descriptor)]
        if len(output_descriptors) == 1 and fc_outputs == output_descriptors:
            converter_context.merge_descriptors(descriptor, fc_outputs[0])
            return

        non_ignored_inputs = [d for d in input_descriptors if not d.is_ignored]
        if len(non_ignored_inputs) == 1:
            tensors = converter_context.get_output_tensors_between(non_ignored_inputs[0], descriptor)
            input_shape = converter_context.graph_helper.get_op_output_shape(tensors[0].op)
            output_shape = converter_context.graph_helper.get_op_output_shape(descriptor.child_ops[0])
            if input_shape == output_shape:
                converter_context.merge_descriptors(descriptor, non_ignored_inputs[0])
        elif len(non_ignored_inputs) == 0:
            descriptor.set_ignored(True)
