#!/usr/bin/env python
#=============================================================================
#
#  Copyright (c) 2018 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
from converters import code_to_message
from converters.tensorflow.util import ConverterError
from converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)
import numpy as np
import snpe


class TileLayerResolver(LayerResolver, object):

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, multiples, output_names=None):
            super(TileLayerResolver.Descriptor, self).__init__('Tile', name, nodes, output_names=output_names)
            self.multiples = multiples

    def __init__(self):
        sequence = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            ConverterSequenceNode('tile', ['Tile']),
            NonConsumableConverterSequenceNode('multiples', ['?'])
        ])
        sequence.set_inputs('tile', ['input', 'multiples'])
        sequence.set_outputs(['tile'])

        self.sequences = [sequence]

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for sequence in self.sequences:
            for match in graph_matcher.match_sequence(sequence):
                tile_op = match['tile']
                multiples_op = match['multiples']
                values = graph_helper.evaluate_tensor_output(multiples_op.outputs[0])

                consumed_nodes = match.consumed_nodes
                tile_descriptor = TileLayerResolver.Descriptor(
                    str(tile_op.name), consumed_nodes, values,
                    output_names=[str(tile_op.outputs[0].name)])
                descriptors.extend([tile_descriptor])

        return descriptors


class TileLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: TileLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return converter_context.model.add_tile_layer(name=descriptor.layer_name,
                                                      multiples=(descriptor.multiples).tolist(),
                                                      input_name=input_name,
                                                      output_name=output_name)
