#==============================================================================
#
#  Copyright (c) 2018 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

from .onnx_translations import *

#------------------------------------------------------------------------------
#   Add
#------------------------------------------------------------------------------
class OnnxAddTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.index_method(SQUASH_SCALE, self.squash_scale)

    def extract_parameters(self, src_op, graph):
        op = op_adapter.ElementwiseSumOp(str(src_op.name))
        # If an Add is broadcast it must come from weights
        if is_broadcast(src_op):
            LOG_WARNING(WARNING_BROADCAST_ADD)
            input_names = list(map(str, src_op.input))
            op.bias = graph.weights.fetch(input_names[1])
        return op

    def extract_input_names(self, src_op, graph):
        if is_broadcast(src_op):
            return [str(src_op.input[0])]
        else:
            return list(map(str, src_op.input))

    def axes_to_snpe_order(self, node, graph):
        eltwise_to_snpe_order(node, graph)

    def squash_scale(self, node, graph):
        if hasattr(node.op, 'bias'):
            input_buffer = graph.get_input_buffers(node)[0]
            prev = input_buffer.producer
            LOG_ASSERT(hasattr(prev.op, 'bias'),
                       ERROR_ADD_BIAS_PREV_NO_BIAS,
                       node.op.name,
                       prev.op.name,
                       prev.op.type)
            prev.op.bias += node.op.bias
            graph.squash(node, input_buffer.name)

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.ElementwiseSumOp.TRANSLATION_KEY]

OnnxTranslations.register(OnnxAddTranslation(),
                          onnx_type('Add'),
                          op_adapter.ElementwiseSumOp.TRANSLATION_KEY)

#------------------------------------------------------------------------------
#   Elu
#------------------------------------------------------------------------------
class OnnxEluTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        # these parameters belong to ScaledTanh
        params = extract_attributes(src_op,
                                    ('alpha','f',1.0))
        return op_adapter.NeuronOp(src_op.name,
                                   modeltools.NEURON_ELU,
                                   a=params.alpha)

    def axes_to_snpe_order(self):
        raise NotImplemented()

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.NeuronOp.TRANSLATION_KEY]

OnnxTranslations.register(OnnxEluTranslation(),
                          onnx_type('Elu'))

#------------------------------------------------------------------------------
#   GEMM
#------------------------------------------------------------------------------
class OnnxGemmTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        LOG_WARNING(WARNING_GEMM)
        params = extract_attributes(src_op,
                                    ('alpha','f',1.0),
                                    ('beta','f',1.0),
                                    ('transA','i',0),
                                    ('transB','i',0))

        LOG_ASSERT(not params.transA,
                   ERROR_GEMM_TRANSPOSE_NOT_SUPPORTED)
        input_names = list(map(str, src_op.input))
        weights, bias = graph.weights.fetch(*input_names[1:])
        weights *= params.alpha
        # for GEMM, weights are supposed to be B and thus KxN.
        # for FC, weights are supposed to be NxK and get transposed
        # implicitly. Transpose explicitly here so that they wind up as NxK
        # for axes_to_snpe_order
        weights = numpy.ascontiguousarray(numpy.transpose(weights, (1,0)))
        bias *= params.beta
        return op_adapter.FullyConnectedOp(src_op.name, [weights], bias)

    def extract_input_names(self, src_op, graph):
        return [str(src_op.input[0])]

    # handled by FC translation
    def axes_to_snpe_order(self):
        raise NotImplemented()

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.FullyConnectedOp.TRANSLATION_KEY]

OnnxTranslations.register(OnnxGemmTranslation(), onnx_type('Gemm'))

#------------------------------------------------------------------------------
#   Matmul
#------------------------------------------------------------------------------
class OnnxMatMulTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        LOG_WARNING(WARNING_MATMUL)
        input_names = list(map(str, src_op.input))
        # SNPE currently only supports FC, so given AxB, B MUST be a set of
        # static weights
        weights = graph.weights.fetch(input_names[1])
        bias = numpy.zeros(weights.shape[1], dtype=numpy.float32)
        return op_adapter.FullyConnectedOp(src_op.name, [weights], bias)

    def extract_input_names(self, src_op, graph):
        return [str(src_op.input[0])]

    # handled by FC translation
    def axes_to_snpe_order(self):
        raise NotImplemented()

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.FullyConnectedOp.TRANSLATION_KEY]

OnnxTranslations.register(OnnxMatMulTranslation(), onnx_type('MatMul'))

#------------------------------------------------------------------------------
#   Max
#------------------------------------------------------------------------------
class OnnxMaxTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        assert_no_broadcast(src_op)
        return op_adapter.ElementwiseMaxOp(str(src_op.name))

    def axes_to_snpe_order(self, node, graph):
        eltwise_to_snpe_order(node, graph)

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.ElementwiseMaxOp.TRANSLATION_KEY]

OnnxTranslations.register(OnnxMaxTranslation(),
                          onnx_type('Max'),
                          op_adapter.ElementwiseMaxOp.TRANSLATION_KEY)

#------------------------------------------------------------------------------
#   Mul
#------------------------------------------------------------------------------
class OnnxMulTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.index_method(SQUASH_SCALE, self.squash_scale)

    def extract_parameters(self, src_op, graph):
        op = op_adapter.ElementwiseProductOp(src_op.name)
        if is_broadcast(src_op):
            LOG_WARNING(WARNING_BROADCAST_MUL)
            input_names = list(map(str, src_op.input))
            op.weights = graph.weights.fetch(input_names[1])
        return op

    def axes_to_snpe_order(self, node, graph):
        eltwise_to_snpe_order(node, graph)

    def extract_input_names(self, src_op, graph):
        if is_broadcast(src_op):
            return [str(src_op.input[0])]
        else:
            return list(map(str, src_op.input))

    def squash_scale(self, node, graph):
        if hasattr(node.op, 'weights'):
            input_buffer = graph.get_input_buffers(node)[0]
            prev = input_buffer.producer
            LOG_ASSERT(prev.op.type == op_adapter.BatchnormOp.TRANSLATION_KEY,
                       ERROR_MUL_SCALE_PREV_NOT_BATCHNORM,
                       prev.op.name,
                       prev.op.type)
            weights = node.op.weights
            prev.op.weights *= weights
            prev.op.bias *= weights
            graph.squash(node, input_buffer.name)

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.ElementwiseProductOp.TRANSLATION_KEY]

OnnxTranslations.register(OnnxMulTranslation(),
                          onnx_type('Mul'),
                          op_adapter.ElementwiseProductOp.TRANSLATION_KEY)

#------------------------------------------------------------------------------
#   Relu
#------------------------------------------------------------------------------
class OnnxReluTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        return op_adapter.NeuronOp(src_op.name, modeltools.NEURON_RELU)

    def axes_to_snpe_order(self, node, graph):
        eltwise_to_snpe_order(node, graph)

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.NeuronOp.TRANSLATION_KEY]

OnnxTranslations.register(OnnxReluTranslation(),
                          onnx_type('Relu'),
                          op_adapter.NeuronOp.TRANSLATION_KEY)

#------------------------------------------------------------------------------
#   Sigmoid
#------------------------------------------------------------------------------
class OnnxSigmoidTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        return op_adapter.NeuronOp(src_op.name, modeltools.NEURON_LOGISTIC, a=1.0)

    # Relu translation already registered to handle neuron op functions.
    def axes_to_snpe_order(self):
        raise NotImplemented()

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.NeuronOp.TRANSLATION_KEY]

OnnxTranslations.register(OnnxSigmoidTranslation(), onnx_type('Sigmoid'))

#------------------------------------------------------------------------------
#   Softmax
#------------------------------------------------------------------------------
class OnnxSoftmaxTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, ('axis','i',1))
        input_buf = graph.get_buffer(str(src_op.input[0]))
        LOG_ASSERT(params.axis == 1,
                   "Node %s: SNPE supports softmax only for axis 1",
                   src_op.name)
        LOG_ASSERT(input_buf.rank() == 2,
                   "Node %s: SNPE supports softmax only for inputs of rank 2",
                   src_op.name)
        return op_adapter.SoftmaxOp(src_op.name)

    def axes_to_snpe_order(self, node, graph):
        # NB will probably want to switch to 'eltwise' version when we
        # support axis parameter.
        feature_to_snpe_order(node, graph)

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.SoftmaxOp.TRANSLATION_KEY]

OnnxTranslations.register(OnnxSoftmaxTranslation(),
                          onnx_type('Softmax'),
                          op_adapter.SoftmaxOp.TRANSLATION_KEY)

#------------------------------------------------------------------------------
#   Sum
#------------------------------------------------------------------------------
class OnnxSumTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseSumOp(src_op.name)

    # OnnxAddTranslation handles the other functions
    def axes_to_snpe_order(self):
        raise NotImplemented()

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.ElementwiseSumOp.TRANSLATION_KEY]

OnnxTranslations.register(OnnxSumTranslation(),onnx_type('Sum'))

#------------------------------------------------------------------------------
#   Tanh, ScaledTanh
#------------------------------------------------------------------------------
class OnnxTanhTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        # these parameters belong to ScaledTanh
        params = extract_attributes(src_op,
                                    ('alpha','f',1.0),
                                    ('beta','f',1.0))
        return op_adapter.NeuronOp(src_op.name,
                                   modeltools.NEURON_TANH,
                                   a=params.alpha,
                                   b=params.beta)

    # OnnxReluTranslation handles the other functions
    def axes_to_snpe_order(self):
        raise NotImplemented()

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.NeuronOp.TRANSLATION_KEY]

OnnxTranslations.register(OnnxTanhTranslation(),
                          onnx_type('Tanh'),
                          onnx_type('ScaledTanh'))
