#!/bin/bash
#==============================================================================
#
#  Copyright (c) 2016 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

# This script sets up the various environment variables needed to run various sdk binaries and scripts
OPTIND=1

_usage()
{
cat << EOF
_usage: $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}) [-h] [-c CAFFE_DIRECTORY] [-f CAFFE2_DIRECTORY] [-t TENSORFLOW_DIRECTORY]

Script sets up environment variables needed for running sdk binaries and scripts, where only one of the
Caffe, Caffe2, or Tensorflow directories have to be specified.

optional arguments:
 -c CAFFE_DIRECTORY            Specifies Caffe directory
 -f CAFFE2_DIRECTORY           Specifies Caffe2 directory
 -o ONNX_DIRECTORY             Specifies ONNX directory
 -t TENSORFLOW_DIRECTORY       Specifies TensorFlow directory

EOF
}

# copy appropriate C++ STL library (libgnustl_shared.so|libc++_shared.so) to lib directory of SNPE SDK
function _copy_stl_lib()
{
  local SNPE_TARGET_DIR=$1 # arm-android-gcc4.9 | aarch64-android-gcc4.9 | arm-android-clang3.8 | aarch64-android-clang3.8
  local ANDROID_NDK_DIR=$2 # armeabi-v7a | arm64-v8a
  local STL_LIB_NAME=$3 # libgnustl_shared.so | libc++_shared.so
  local AAR_NAME=$4 # snpe-gcc-release.aar | snpe-release.aar
  local STL_LIB_VERSION=$5 # Optional: 4.9

  local CXX_STL_COPY=
  if [ ! -e "$SNPE_ROOT/lib/$SNPE_TARGET_DIR/$STL_LIB_NAME" ]; then
    echo "[WARNING] Cannot find "$SNPE_ROOT/lib/$SNPE_TARGET_DIR/$STL_LIB_NAME

    if [[ -d "$ANDROID_NDK_ROOT" ]]; then
      local FOUND_CXX_STL=
      if [ ! -z $STL_LIB_VERSION ]; then
        FOUND_CXX_STL=`find $ANDROID_NDK_ROOT/ -name $STL_LIB_NAME | grep "$STL_LIB_VERSION" | grep "$ANDROID_NDK_DIR\/$STL_LIB_NAME"`
      else
        FOUND_CXX_STL=`find $ANDROID_NDK_ROOT/ -name $STL_LIB_NAME $GREP_VERSION | grep "$ANDROID_NDK_DIR\/$STL_LIB_NAME"`
      fi

      echo "[INFO] $STL_LIB_NAME found at "$FOUND_CXX_STL
      echo "[INFO] Copying $STL_LIB_NAME to $SNPE_ROOT/lib/$SNPE_TARGET_DIR"

      cp $FOUND_CXX_STL $SNPE_ROOT/lib/$SNPE_TARGET_DIR
      CXX_STL_COPY=1
    else
      echo "[WARNING] Please copy $STL_LIB_NAME for $ANDROID_NDK_DIR from the Android NDK into ${SNPE_ROOT}/lib/$SNPE_TARGET_DIR"
    fi
  else
    echo "[INFO] Found $SNPE_ROOT/lib/$SNPE_TARGET_DIR/$STL_LIB_NAME"
    CXX_STL_COPY=1
  fi

  if [[ $CXX_STL_COPY ]]
  then
      echo "[INFO] Adding $ANDROID_NDK_DIR $STL_LIB_NAME to $SNPE_ROOT/android/$AAR_NAME"
      pushd $SNPE_ROOT/android > /dev/null
      mkdir -p jni/$ANDROID_NDK_DIR/
      cp $SNPE_ROOT/lib/$SNPE_TARGET_DIR/$STL_LIB_NAME jni/$ANDROID_NDK_DIR/
      zip -rq $AAR_NAME jni/$ANDROID_NDK_DIR/$STL_LIB_NAME
      rm -rf jni
      popd > /dev/null
  fi
}

function _setup_snpe()
{
  # get directory of the bash script
  local SOURCEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
  export SNPE_ROOT=$(readlink -f $SOURCEDIR/..)
  export PATH=$SNPE_ROOT/bin/x86_64-linux-clang:$PATH

  # setup LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$SNPE_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH

  # setup PYTHONPATH
  export PYTHONPATH=$SNPE_ROOT/lib/python:$PYTHONPATH
  export PYTHONPATH=$SNPE_ROOT/models/lenet/scripts:$PYTHONPATH
  export PYTHONPATH=$SNPE_ROOT/models/alexnet/scripts:$PYTHONPATH
}

function _setup_caffe()
{
  if ! _is_valid_directory $1; then
    return 1
  fi

  # common setup
  _setup_snpe

  local CAFFEDIR=$1

  # current tested SHA for caffe
  local VERIFY_CAFFE_SHA="d8f79537977f9dbcc2b7054a9e95be00eb6f26d0"

  # setup an environment variable called $CAFFE_HOME
  export CAFFE_HOME=$CAFFEDIR
  echo "[INFO] Setting CAFFE_HOME="$CAFFEDIR

  # update PATH
  export PATH=$CAFFEDIR/build/install/bin:$PATH
  export PATH=$CAFFEDIR/distribute/bin:$PATH

  # update LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$CAFFEDIR/build/install/lib:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CAFFEDIR/distribute/lib

  # update PYTHONPATH
  export PYTHONPATH=$CAFFEDIR/build/install/python:$PYTHONPATH
  export PYTHONPATH=$CAFFEDIR/distribute/python:$PYTHONPATH

  # check Caffe SHA
  pushd $CAFFEDIR > /dev/null
  local CURRENT_CAFFE_SHA=$(git rev-parse HEAD)
  if [ "$VERIFY_CAFFE_SHA" != "$CURRENT_CAFFE_SHA" ]; then
    echo "[WARNING] Expected CAFFE HEAD rev "$VERIFY_CAFFE_SHA" but found "$CURRENT_CAFFE_SHA" instead. This SHA is not tested."
  fi
  popd > /dev/null

  return 0
}

function _setup_caffe2()
{
  if ! _is_valid_directory $1; then
    return 1
  fi

  # common setup
  _setup_snpe

  local CAFFE2DIR=$1

  # setup an environment variable called $CAFFE2_HOME
  export CAFFE2_HOME=$CAFFE2DIR
  echo "[INFO] Setting CAFFE2_HOME="$CAFFE2DIR

  # setup LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

  # setup PYTHONPATH
  export PYTHONPATH=$CAFFE2DIR/build/:$PYTHONPATH
  export PYTHONPATH=/usr/local/:$PYTHONPATH

  return 0
}

function _setup_onnx()
{
  if ! _is_valid_directory $1; then
    return 1
  fi

  # common setup
  _setup_snpe

  local ONNXDIR=$1

  # setup an environment variable called $ONNX_HOME
  export ONNX_HOME=$ONNXDIR
  echo "[INFO] Setting ONNX_HOME="$ONNXDIR

  return 0
}

function _setup_tensorflow()
{
  if ! _is_valid_directory $1; then
    return 1
  fi

  # common setup
  _setup_snpe

  local TENSORFLOWDIR=$1

  # setup an environment variable called $TENSORFLOW_HOME
  export TENSORFLOW_HOME=$TENSORFLOWDIR
  echo "[INFO] Setting TENSORFLOW_HOME="$TENSORFLOWDIR

  return 0
}

function _check_ndk()
{
  # check NDK in path
  if [[ ! -d "$ANDROID_NDK_ROOT" ]]; then
    local ndkDir=$(which ndk-build)
    if [ ! -s "$ndkDir" ]; then
      echo "[WARNING] Can't find ANDROID_NDK_ROOT or ndk-build. SNPE needs android ndk to build the NativeCppExample"
    else
      ANDROID_NDK_ROOT=$(dirname $ndkDir)
      echo "[INFO] Found ndk-build at " $ndkDir
    fi
  else
    echo "[INFO] Found ANDROID_NDK_ROOT at "$ANDROID_NDK_ROOT
  fi
}

function _is_valid_directory()
{
  if [[ ! -z $1 ]]; then
    if [[ ! -d $1 ]]; then
      echo "[ERROR] Invalid directory "$1" specified. Please rerun the srcipt with a valid directory path."
      return 1
    else
      return 0
    fi
  else
    return 1
  fi
}

function _cleanup()
{
  unset -f _usage
  unset -f _copy_libgnustl
  unset -f _setup_snpe
  unset -f _setup_caffe
  unset -f _setup_caffe2
  unset -f _setup_tensorflow
  unset -f _check_ndk
  unset -f _is_valid_directory
  unset -f _cleanup
}

# script can only handle one framework per execution
[[ ($# -le 2) && ($# -gt 0) ]] || { echo "[ERROR] Invalid number of arguments. See -h for help."; return 1; }
_setup_snpe
# parse arguments
while getopts "h?c:f:o:t:" opt; do
  case $opt in
    h  ) _usage; return 0 ;;
    c  ) _setup_caffe $OPTARG || return 1 ;;
    f  ) _setup_caffe2 $OPTARG || return 1 ;;
    o  ) _setup_onnx $OPTARG || return 1 ;;
    t  ) _setup_tensorflow $OPTARG || return 1 ;;
    \? ) return 1 ;;
  esac
done

# check for NDK
_check_ndk

# Make sure libgnustl_shared.so has been copied over to the arm-android-gcc4.9 and aarch64-android-gcc4.9 directory
_copy_stl_lib "arm-android-gcc4.9" "armeabi-v7a" "libgnustl_shared.so" "snpe-gcc-release.aar" "4.9"
_copy_stl_lib "aarch64-android-gcc4.9" "arm64-v8a" "libgnustl_shared.so" "snpe-gcc-release.aar" "4.9"

# cleanup
_cleanup
