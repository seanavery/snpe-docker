#==============================================================================
#
#  Copyright (c) 2018 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================
import numpy
from .. import translation, op_adapter
from ..op_graph import AxisFormat
from snpe import modeltools
from .util import *

OP_VERSION_SUPPORTED = {
    'input': [1],
    'batchnorm': [1, 6, 7],
    'convolution': [1],
    'concatenation': [1, 4],
    'constant': [1],
    'crop': [1],
    'deconvolution': [1],
    'elementwise_max': [1, 6, 8],
    'elementwise_product': [1, 6, 7],
    'elementwise_sum': [1, 6, 7],
    'fully_connected': [1],  # Handles FC, GEMM and MatMul. Ignored GEMM op set until it's support is there.
    'neuron': [1, 6],  # Handles Clip, Relu, Sigmoid, Tanh , and Elu operations for now.
    'pad': [1, 2],
    'pool': [1],
    'permute': [1],
    'prelu': [1, 6, 7],
    'reshape': [1, 5],  # Handles Flatten {1} and Reshape {1, 5} operations. Used the larger set for now.
    'rnorm': [1],
    'roi_pooling': [1],
    'resize': [1],  # TO_DO
    'shape': [1],
    'slice': [1],
    'squeeze':[1],
    'softmax': [1],
    'unsqueeze':[1]
}

OnnxTranslations = translation.TranslationBank()
ADD_OP = "ADD_OP"
INFER_SHAPE = "INFER_SHAPE"
REMOVE_NOOP = "REMOVE_NOOP"
SQUASH_BATCHNORM = "SQUASH_BATCHNORM"
SQUASH_SCALE = "SQUASH_SCALE"
AXES_TO_SNPE_ORDER = "AXES_TO_SNPE_ORDER"
SUPPORTED_VERSION = "SUPPORTED_VERSION"


def inject_implicit_permute(graph, input_name, target_format, permute_order, consumers=None):
    permute_name = input_name +'.'+target_format.lower()
    implicit_permute = op_adapter.PermuteOp(permute_name, permute_order)
    graph.inject(implicit_permute, input_name, permute_name, consumers)
    # since the implicit permute won't be visited in this pass, go
    # ahead and set the correct order for its buffer here.
    permute_buf = graph.get_buffer(permute_name)
    permute_buf.axis_format = target_format

def enforce_input_format(node, graph, input_name, target_format, permute_order):
    input_buf = graph.get_buffer(input_name)
    if input_buf.axis_format == AxisFormat.NONTRIVIAL:
        inject_implicit_permute(graph, input_name, target_format, permute_order)
    elif input_buf.axis_format != target_format:
        raise ValueError(ERROR_INPUT_DATA_ORDER_UNEXPECTED.format(name,
                                                                  target_format,
                                                                  input_buf.axis_format))

def permute_shape(shape, order):
    return [ shape[i] for i in order ]

# well-known permute orders
NCS_TO_NSC = [0,2,3,1]
NSC_TO_NCS = [0,3,1,2]
TBF_TO_BTF = [1,0,2]

def image_to_snpe_order(node, graph):
    """Axis transformation for layers which take in and emit only image-valued data"""

    # (1) if any of our inputs are NONTRIVIAL, put a permute
    # of NCS -> NSC in front of them. This will be shared
    # with everyone who consumes that buffer, so don't specify consumers
    for name in node.input_names:
        # fetch input buffers one by one to avoid degenerate case where
        # an op uses the same input more than once and needs to permute it.
        enforce_input_format(node, graph, name, AxisFormat.NSC, NCS_TO_NSC)

    # (2) Update all of our output buffers to be in NSC order.
    for buf in graph.get_output_buffers(node):
        buf.shape = permute_shape(buf.shape, NCS_TO_NSC)
        buf.axis_format = AxisFormat.NSC

def feature_to_snpe_order(node, graph):
    # Not much to do here, just mark the outputs
    for buf in graph.get_output_buffers(node):
        buf.axis_format = AxisFormat.FEATURE

def time_series_to_snpe_order(node, graph):
    for name in node.input_names:
        enforce_input_format(node, graph, name, AxisFormat.BTF, TBF_TO_BTF)

    for buf in graph.get_output_buffers(node):
        buf.shape = permute_shape(buf.shape, TBF_TO_BTF)
        buf.axis_format = AxisFormat.BTF

def eltwise_to_snpe_order(node, graph):
    input_buffers = graph.get_input_buffers(node)
    input_orders = [buf.axis_format for buf in input_buffers]
    if AxisFormat.NSC in input_orders:
        image_to_snpe_order(node, graph)
    elif AxisFormat.FEATURE in input_orders:
        feature_to_snpe_order(node, graph)
    elif AxisFormat.BTF in input_orders:
        time_series_to_snpe_order(node, graph)
    else:
        # well hopefully someone knows
        for buf in graph.get_output_buffers(node):
            buf.axis_format = AxisFormat.NONTRIVIAL

def log_axes_to_snpe_order(node, graph):
    LOG_DEBUG(DEBUG_AXES_TO_SNPE_ORDER_ENTRY, node.op.name)
    for input_name in node.input_names:
        LOG_DEBUG(DEBUG_AXES_TO_SNPE_ORDER_INPUT_SIZE,
                  input_name,
                  str(graph.get_buffer(input_name).shape))

class OnnxTranslationBase(translation.Translation):
    def __init__(self):
        translation.Translation.__init__(self)
        self.index_method(ADD_OP, self.add_op)
        self.index_method(INFER_SHAPE, self.infer_output_shapes)
        self.index_method(AXES_TO_SNPE_ORDER, self.axes_to_snpe_order)
        self.index_method(SUPPORTED_VERSION, self.supported_version)

    def add_op(self, src_op, graph):
        op = self.extract_parameters(src_op, graph)
        input_names = self.extract_input_names(src_op, graph)
        output_names = self.extract_output_names(src_op, graph)
        graph.add(op, input_names, output_names)

    def extract_input_names(self, src_op, graph):
        return list(map(str, src_op.input))

    def extract_output_names(self, src_op, graph):
        return list(map(str, src_op.output))

    def infer_output_shapes(self, node, input_shapes):
        return [input_shapes[0]]

    def supported_version(self):
        return self.get_supported_version()

#------------------------------------------------------------------------------
#   StaticOp
#------------------------------------------------------------------------------
# 'Static' ops are transformations applied to weights, which do not produce
# an actual runtime output.
class OnnxStaticOp(op_adapter.Op):
    TRANSLATION_KEY = 'static'
    def __init__(self, name):
        op_adapter.Op.__init__(self, name, self.TRANSLATION_KEY)

class OnnxStaticTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.index_method(REMOVE_NOOP, self.remove_noop)

    def infer_output_shapes(self, op, input_shapes):
        return []

    def axes_to_snpe_order(self, node, graph):
        pass

    def remove_noop(self, node, graph):
        graph.prune(node)

OnnxTranslations.register(OnnxStaticTranslation(), OnnxStaticOp.TRANSLATION_KEY)

#------------------------------------------------------------------------------
#   Input
#------------------------------------------------------------------------------
# ONNX doesn't have an input layer, but we need to handle the later stages
# of processing for SNPE.
class OnnxInputTranslation(OnnxTranslationBase):
    def axes_to_snpe_order(self, node, graph):
        buf = graph.get_buffer(node.output_names[0])
        if node.op.image_type == 'opaque':
            buf.axis_format = AxisFormat.NONTRIVIAL
        elif buf.rank() == 4:
            buf.shape = permute_shape(buf.shape, NCS_TO_NSC)
            buf.axis_format = AxisFormat.NSC
            node.op.shape = buf.shape
        elif buf.rank() == 2:
            buf.axis_format = AxisFormat.FEATURE
            node.op.shape = buf.shape
        else:
            raise ValueError(ERROR_INPUT_UNEXPECTED_RANK.format(node.op.name, buf.rank()))

OnnxTranslations.register(OnnxInputTranslation(),
                          op_adapter.InputOp.TRANSLATION_KEY)

#------------------------------------------------------------------------------
#   Dropout, and other Noops
#------------------------------------------------------------------------------
class OnnxNoop(op_adapter.Op):
    TRANSLATION_KEY = 'noop'
    def __init__(self, name):
        op_adapter.Op.__init__(self, name, self.TRANSLATION_KEY)


class OnnxNoopTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.index_method(REMOVE_NOOP, self.remove_noop)

    def extract_parameters(self, src_op, graph):
        return OnnxNoop(src_op.name)

    def extract_output_names(self, src_op, graph):
        return [str(src_op.output[0])]

    def remove_noop(self, node, graph):
        graph.squash(node, node.input_names[0])

    def axes_to_snpe_order(self, node, graph):
        output_buf = graph.get_output_buffers(node)[0]
        input_buf = graph.get_input_buffers(node)[0]
        output_buf.shape = input_buf.shape
        output_buf.axis_format = input_buf.axis_format

    def get_supported_version(self):
        return {}

OnnxTranslations.register(OnnxNoopTranslation(),
                          onnx_type('Dropout'),
                          OnnxNoop.TRANSLATION_KEY)

#------------------------------------------------------------------------------
#   Class OpVersionInfo
#------------------------------------------------------------------------------
# Returns name and version information about an op from a particular model
class OpVersionInfo():
    model_opset_version = 0
    def __init__(self):
        self.op_version_dict = dict()
        self.setup_op_version_dict()

    def setup_op_version_dict(self):
        for schema in defs.get_all_schemas_with_history():
            # Splitting the operator name and storing the version in op_version_dict
            self.op_version_dict[op_type(schema.name)] = schema.since_version

    def get_op_ver_dict(self):
        return self.op_version_dict

    def validate_op_ver(self, src_op, supported_version):
        if self.op_version_dict[op_type(src_op.op_type)] not in supported_version:
            LOG_WARNING(WARNING_OP_NOT_SUPPORTED, src_op.op_type)

    def set_global_op_ver(self, model):
        """ Sets the highest global op version supported by the model"""
        # Get the global opset version
        if len(model.opset_import) > 1:
            LOG_WARNING(WARNING_OPSET_VERION)

        for opset in model.opset_import:
            if opset.version > OpVersionInfo.model_opset_version:
                OpVersionInfo.model_opset_version = opset.version

    @staticmethod
    def onnx_op_ver(src_op, supported_version):
        """Return the actual op version. If embedded in the op name return that,
           otherwise get the global op version and correlate to the highest op version
           supported as per the onnx.proto specification"""
        onnx_data =  get_op_info(src_op.op_type)
        # If op is missing version, use the version as the minimum of the supported
        # model opset version and the largest supported op version in the converter
        # TODO See if there is a way to lookup the current op version information for
        # a given model_opset_version... this is really what we should be using instead
        # of the actual model_opset_verison
        if onnx_data[1] == 0:
            min_supported_version = min(supported_version[-1], OpVersionInfo.model_opset_version)
            return onnx_data[0], min_supported_version
        return onnx_data[0], onnx_data[1]

