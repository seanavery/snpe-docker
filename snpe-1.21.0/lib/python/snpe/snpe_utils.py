#!/usr/bin/env python
# -*- mode: python -*-
#=============================================================================
#
#  Copyright (c) 2018 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================

import logging
import numpy

class SNPEUtils(object):
    def blob2arr(self, blob):
        if hasattr(blob, "shape"):
            return numpy.ndarray(buffer=blob.data, shape=blob.shape, dtype=numpy.float32)
        else:
       #Caffe-Segnet fork doesn't have shape field exposed on blob.
            return numpy.ndarray(buffer=blob.data, shape=blob.data.shape, dtype=numpy.float32)

    def setUpLogger(self, verbose):
        formatter = '%(asctime)s - %(lineno)d - %(levelname)s - %(message)s'
        lvl = logging.INFO
        if verbose:
             lvl = logging.DEBUG
        logger = logging.getLogger()
        logger.setLevel(lvl)
        formatter = logging.Formatter(formatter)

        ch = logging.StreamHandler()
        ch.setLevel(lvl)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
