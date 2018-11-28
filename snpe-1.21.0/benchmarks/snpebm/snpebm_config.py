import json
import os
from snpebm_jsonkeys import *
from snpebm_config_restrictions import *
from android_utils import get_device_list
import datetime
from subprocess import check_output
import numpy as np

class ConfigError(Exception):
    def __str__(self):
        return ('\nConfiguration Error: ' + repr(self) + '\n')


def load_json(cfgfile, logger):
    try:
        with open(cfgfile, 'r') as cfg:
            try:
                json_data = json.load(cfg)
                return json_data
            except ValueError as e:
                logger.error("error parsing JSON file: " + cfgfile)
                return []
    except Exception as e:
        logger.error("Error opening file: " + cfgfile + " " + repr(e))
        return []


class DnnModel:
    def __init__(self, config, host_artifacts):
        self._name = config[CONFIG_MODEL_KEY][CONFIG_MODEL_NAME_SUBKEY]
        self._dlc = DnnModel.__default_path(config[CONFIG_MODEL_KEY][CONFIG_MODEL_DLC_SUBKEY])
        self._dev_root_dir = os.path.join(config[CONFIG_DEVICE_PATH_KEY], self._name)
        self._host_artifacts = host_artifacts
        subkeys = []
        for sub_key in config[CONFIG_MODEL_KEY]:
            subkeys.append(sub_key)
        if "InputList" in subkeys:
            self._input_list_name = os.path.basename(config[CONFIG_MODEL_KEY][CONFIG_MODEL_INPUTLIST_SUBKEY])
            self._artifacts = []
            self._artifacts.append([self._dlc, self._dev_root_dir])
            self._artifacts.append([DnnModel.__default_path(config[CONFIG_MODEL_KEY][CONFIG_MODEL_INPUTLIST_SUBKEY]),
                                    self._dev_root_dir])
            for data in config[CONFIG_MODEL_KEY][CONFIG_MODEL_DATA_SUBKEY]:
                _abs_data_path = DnnModel.__default_path(data)
                self._artifacts.append([_abs_data_path, os.path.join(self._dev_root_dir, os.path.basename(data))])
        elif "RandomInput" in subkeys:
            # Get the input dimensions and generate random data
            input_layer_dim = self.input_layers
            num_inputs = config[CONFIG_MODEL_KEY][CONFIG_MODEL_RANDOMINPUT_SUBKEY]
            if num_inputs < 1:
                raise ConfigError('Input can not be less than 1')
            random_data_dim_str = input_layer_dim[0].split(":")[-1]
            random_data_dim = random_data_dim_str.split(",")
            random_data_dim = map(int, random_data_dim)
            random_inputs_dir = os.path.join(config[CONFIG_HOST_ROOTPATH_KEY], 'random_inputs')
            _abs_random_inputs_path = os.path.abspath(random_inputs_dir)
            if not os.path.isdir(_abs_random_inputs_path):
                os.makedirs(_abs_random_inputs_path)
            self._input_list_name = "random_raw_list.txt"
            input_file_path = os.path.join(_abs_random_inputs_path, self._input_list_name)
            input_file = open(input_file_path, 'w')
            for i in range (1, num_inputs+1):
                raw_filepath = os.path.join(_abs_random_inputs_path, 'random_input_' + str(i) + '.raw')
                rand_raw = np.random.uniform(-1.0, +1.0, random_data_dim).astype(np.float32)
                with open(raw_filepath, 'wb') as fid:
                    fid.write(rand_raw)
                raw_rel_path = os.path.basename(random_inputs_dir) + "/" + os.path.basename(raw_filepath)
                input_file.write(raw_rel_path + "\n")
            input_file.close()
            self._artifacts = []
            self._artifacts.append([self._dlc, self._dev_root_dir])
            self._artifacts.append([input_file_path, self._dev_root_dir])
            self._artifacts.append([_abs_random_inputs_path, os.path.join(self._dev_root_dir, os.path.basename(_abs_random_inputs_path))])

    @staticmethod
    def __default_path(artifact_path):
        _abs_path = artifact_path
        if not os.path.isabs(artifact_path):
            # relative to current directory
            _abs_path = os.path.abspath(artifact_path)
        if not os.path.exists(_abs_path):
            raise ConfigError(artifact_path + " does not exist")
        return _abs_path

    @property
    def name(self):
        return self._name

    @property
    def input_list_name(self):
        return self._input_list_name

    @property
    def dlc(self):
        return self._dlc

    @property
    def device_rootdir(self):
        return self._dev_root_dir

    @property
    def artifacts(self):
        return self._artifacts

    @property
    def input_layers(self):
        dlc_cmd = [self._host_artifacts[SNPE_DLC_INFO_EXE], '-i', self._dlc]
        try:
            dlc_info_output = check_output(dlc_cmd).decode()
        except Exception as de:
            self._logger.warning("Failed to parse {0}".format(self._dlc))
            self._logger.warning(repr(de))
            return []

        inputs = []
        for line in dlc_info_output.split('\n'):
            if ("------------------" in line) or ("Id" in line) or ("Training" in line) or ("Concepts" in line):
                continue
            split_line = line.replace(" ", "").split("|")
            if len(split_line) > 6 and split_line[3] == "data":
                layer_name = split_line[2]
                input_dimensions = split_line[6].replace("x", ",")
                inputs.append("{}:{}".format(layer_name, input_dimensions))
        return inputs


class Config:
    def __init__(self, cfg_from_json, outputbasedir, devicelist, hostname, deviceostype, logger, userbuffer_mode, perfprofile):
        self._logger = logger
        self._cfg_from_json = cfg_from_json
        if deviceostype and deviceostype not in CONFIG_VALID_DEVICEOSTYPES:
            raise ConfigError('Device OS Type not valid.  Only specify one of %s'%CONFIG_VALID_DEVICEOSTYPES)
        if deviceostype == CONFIG_DEVICEOSTYPES_LE:
            self._architectures = [ARCH_ARM]
            self._platform_os = PLATFORM_OS_LINUX
            self._compiler = COMPILER_GCC49
            self._stl_library = STL_GNUSTL_SHARED
            self._artifacts_config = load_json(os.path.join("snpebm", SNPE_BENCH_LE_ARTIFACTS_JSON), self._logger)
        elif deviceostype == CONFIG_DEVICEOSTYPES_LE64_GCC49:
            self._architectures = [ARCH_AARCH64]
            self._platform_os = PLATFORM_OS_LINUX
            self._compiler = COMPILER_GCC49
            self._stl_library = STL_GNUSTL_SHARED
            self._artifacts_config = load_json(os.path.join("snpebm", SNPE_BENCH_LE64_GCC49_ARTIFACTS_JSON), self._logger)
        elif deviceostype == CONFIG_DEVICEOSTYPES_LE64_GCC53:
            self._architectures = [ARCH_AARCH64]
            self._platform_os = PLATFORM_OS_LINUX
            self._compiler = COMPILER_GCC53
            self._stl_library = STL_GNUSTL_SHARED
            self._artifacts_config = load_json(os.path.join("snpebm", SNPE_BENCH_LE64_GCC53_ARTIFACTS_JSON), self._logger)
        elif deviceostype == CONFIG_DEVICEOSTYPES_ANDROID_AARCH64:
            self._architectures = [ARCH_AARCH64]
            self._compiler = COMPILER_GCC49
            self._platform_os = PLATFORM_OS_ANDROID
            self._stl_library = STL_GNUSTL_SHARED
            self._artifacts_config = load_json(os.path.join("snpebm", SNPE_BENCH_ANDROID_AARCH64_ARTIFACTS_JSON), self._logger)
        elif deviceostype == CONFIG_DEVICEOSTYPES_ANDROID_ARM32_LLVM:
            self._architectures = [ARCH_ARM]
            self._platform_os = PLATFORM_OS_ANDROID
            self._compiler = COMPILER_CLANG38
            self._stl_library = STL_LIBCXX_SHARED
            self._artifacts_config = load_json(os.path.join("snpebm", SNPE_BENCH_ANDROID_ARM32_LLVM_ARTIFACTS_JSON), self._logger)
        elif deviceostype == CONFIG_DEVICEOSTYPES_ANDROID_AARCH64_LLVM:
            self._architectures = [ARCH_AARCH64]
            self._platform_os = PLATFORM_OS_ANDROID
            self._compiler = COMPILER_CLANG38
            self._stl_library = STL_LIBCXX_SHARED
            self._artifacts_config = load_json(os.path.join("snpebm", SNPE_BENCH_ANDROID_AARCH64_LLVM_ARTIFACTS_JSON), self._logger)
        elif deviceostype == CONFIG_DEVICEOSTYPES_LE_GCC48_HF:
            self._architectures = [ARCH_ARM]
            self._platform_os = PLATFORM_OS_LINUX
            self._compiler = COMPILER_GCC48
            self._stl_library = STL_GNUSTL_SHARED
            self._artifacts_config = load_json(os.path.join("snpebm", SNPE_BENCH_LE_GCC48_HF_ARTIFACTS_JSON), self._logger)
        elif deviceostype:
            # presumed to be Android arm32
            self._architectures = [ARCH_ARM]
            self._platform_os = PLATFORM_OS_ANDROID
            self._compiler = COMPILER_GCC49
            self._stl_library = STL_GNUSTL_SHARED
            self._artifacts_config = load_json(os.path.join("snpebm", SNPE_BENCH_ANDROID_ARM32_ARTIFACTS_JSON), self._logger)
        self.__override_cfgfile__(outputbasedir, devicelist, hostname, logger)
        self.__quick_verify__()
        self._dnnmodel = DnnModel(self._cfg_from_json, self.host_artifacts)

        self.userbuffer_mode = userbuffer_mode
        try:
            self.hostname = self._cfg_from_json[CONFIG_HOST_NAME_KEY]
        except KeyError:
            self._cfg_from_json[CONFIG_HOST_NAME_KEY] = 'localhost'

        try:
            self.cpu_fallback = self._cfg_from_json[CONFIG_CPU_FALLBACK_KEY]
        except KeyError:
            self.cpu_fallback = False

        try:
            if not self._cfg_from_json[CONFIG_PERF_PROFILE_KEY] is 'null':
                self.perfprofile = self._cfg_from_json[CONFIG_PERF_PROFILE_KEY]
        except KeyError as ke:
            self.perfprofile = perfprofile

        try:
            self.buffertypes = self._cfg_from_json[CONFIG_BUFFERTYPES_KEY]
        except KeyError:
            self.buffertypes = None

        # Required to set it to default 'high_performance' in case
        # perf_profile is not provided as runtime argument or in the config file
        if self.perfprofile == '' and perfprofile == '':
            self.perfprofile = 'high_performance'
        # Required to give preference to runtime argument in case
        # perf_profile is provided both as runtime argument and in the config file
        elif perfprofile != '':
            self.perfprofile = perfprofile


    def __override_cfgfile__(self, outputbasedir, devicelist, hostname, logger):
        # Override output base dir if the host paths are relative paths
        # after override, it will become an absolute path
        if outputbasedir is not None:
            logger.info('Overriding output base dir to %s, instead of %s' % (outputbasedir, os.getcwd()))
            for _key in [CONFIG_HOST_ROOTPATH_KEY, CONFIG_HOST_RESULTSDIR_KEY]:
                _value = self._cfg_from_json[_key]
                if not os.path.isabs(_value):
                    _abs_path = os.path.abspath(outputbasedir+'/'+_value)
                    self._cfg_from_json[_key] = _abs_path
        # Override device id if one's supplied
        if devicelist is not None:
            _prevDevices = self._cfg_from_json[CONFIG_DEVICES_KEY]
            self._cfg_from_json[CONFIG_DEVICES_KEY] = devicelist
            logger.info('Overriding device id to %s, instead of %s from config file' % (self._cfg_from_json[CONFIG_DEVICES_KEY], _prevDevices))
        # Override Host Name is one's supplied
        if hostname is not None:
            self._cfg_from_json[CONFIG_HOST_NAME_KEY] = hostname
            logger.info('Overriding host name to %s' % (self._cfg_from_json[CONFIG_HOST_NAME_KEY]))

    def __quick_verify__(self):
        try:
            # Check that we have at least all top level keys
            # note that this will give an exception on any key that
            # isn't present
            # NOTE: We do not validate the values provided for that key
            for _key in CONFIG_JSON_ROOTKEYS:
                # check optional keys
                if not _key is CONFIG_PERF_PROFILE_KEY and not _key is CONFIG_CPU_FALLBACK_KEY and not _key is CONFIG_HOST_NAME_KEY and not _key is CONFIG_BUFFERTYPES_KEY:
                    if self._cfg_from_json[_key] is 'null':
                        raise ConfigError("Missing value for " + _key)
            # Check for no foreign top level keys
            for _key in self._cfg_from_json.keys():
                if _key not in CONFIG_JSON_ROOTKEYS:
                    raise ConfigError("Found invalid top level key: " + _key)
        except KeyError as ke:
            raise ConfigError('Missing key in config file: ' + repr(ke))
        # We have all top level keys.  Check some of the sub keys
        if 'RandomInput' in self._cfg_from_json[CONFIG_MODEL_KEY]:
            subkeys = CONFIG_JSON_MODEL_COMMON_SUBKEYS + CONFIG_JSON_MODEL_RANDOM_INPUT_SUBKEYS
            for _key in subkeys:
                if not _key in self._cfg_from_json[CONFIG_MODEL_KEY]:
                    raise ConfigError("No " + CONFIG_MODEL_KEY + ":" + _key + " found")
        else:
            subkeys = CONFIG_JSON_MODEL_COMMON_SUBKEYS + CONFIG_JSON_MODEL_DEFINED_INPUT_SUBKEYS
            for _key in subkeys:
                if not _key in self._cfg_from_json[CONFIG_MODEL_KEY]:
                    raise ConfigError("No " + CONFIG_MODEL_KEY + ":" + _key + " found")

        # All relative paths are relative to the current directory
        for _key in [CONFIG_HOST_ROOTPATH_KEY, CONFIG_HOST_RESULTSDIR_KEY]:
            _value = self._cfg_from_json[_key]
            if not os.path.isabs(_value):
                _abs_path = os.path.abspath(_value)
                if os.path.isfile(_abs_path):
                    raise ConfigError(_key + " is not a directory")
                self._cfg_from_json[_key] = os.path.abspath(_value)

        # Check artifacts paths, relative path is not supported
        # currently hardcoded to ARM as target architecture
        for _arch in self.architectures:
            for _compiler, _artifacts in list(self._artifacts_config[CONFIG_ARTIFACTS_KEY].items()):
                if (not _compiler.startswith(_arch)) and (not _compiler.startswith(ARCH_X86) and (not _compiler.startswith(ARCH_DSP) )):
                    continue
                for _artifact_path in _artifacts:
                    if (not os.path.isabs(_artifact_path)) and os.path.dirname(_artifact_path):
                        raise ConfigError("{0} does not support relative path".format(CONFIG_ARTIFACTS_KEY))
                    elif os.path.isabs(_artifact_path) and (not os.path.exists(_artifact_path)):
                        raise ConfigError("{0} does not exist".format(_artifact_path))
                    elif not os.path.dirname(_artifact_path):
                        if not os.path.exists(self.__default_artifact_path(_compiler, _artifact_path)):
                            raise ConfigError("Could not find {0} for {1}, path used {2}".format(_artifact_path, _compiler, self.__default_artifact_path(_compiler, _artifact_path)))
        # at the momemnt, despite devices takes in a list, benchmark does not work correctly when multiple devices are specified
        device_list = self._cfg_from_json.get(CONFIG_DEVICES_KEY, None)
        if len(device_list) == 0 or not device_list[0]:
            raise ConfigError('Benchmark does not have any device specified')
        elif len(device_list) != 1:
            raise ConfigError('Benchmark does not yet support more than 1 device')

        # Measurements allowed are "timing" and "mem"
        if  0 == len(self._cfg_from_json.get(CONFIG_MEASUREMENTS_KEY, None)):
            raise ConfigError('Benchmark does not specify what to measure')
        else:
            for measurement in self._cfg_from_json.get(CONFIG_MEASUREMENTS_KEY, None):
                if measurement not in [MEASURE_TIMING, MEASURE_MEM]:
                    raise ConfigError('"%s" is unknown measurement'%measurement)


    @staticmethod
    def __default_artifact_path(compiler, artifact):
        # first check for ZDL_ROOT, because if you have both, contributor role takes precedence
        if ZDL_ROOT not in os.environ:
            if SNPE_SDK_ROOT not in os.environ:
                raise ConfigError("Environment variables 'SNPE_ROOT' and 'ZDL_ROOT' are not defined, absolute path is needed for " + artifact + " in snpebm_artifacts.json")
            _sdk_root = os.environ[SNPE_SDK_ROOT]
            _base_name = os.path.basename(artifact)
            if _base_name.endswith(".so") or _base_name.startswith("lib"):
                return os.path.join(_sdk_root, "lib", compiler, artifact)
            else:
                return os.path.join(_sdk_root, "bin", compiler, artifact)
        else:
            _zdl_root = os.environ[ZDL_ROOT]
            _base_name = os.path.basename(artifact)
            if _base_name.endswith(".so") or _base_name.startswith("lib"):
                return os.path.join(_zdl_root, compiler, "lib", artifact)
            else:
                return os.path.join(_zdl_root, compiler, "bin", artifact)

    def measurement_types_are_valid(self):
        for item in self.measurements:
            if item not in CONFIG_VALID_MEASURMENTS:
                return False
        return True


    @property
    def csvrows(self):
        _csvrows = []
        _csvrows.append([CONFIG_NAME_KEY] + [self.name])
        _csvrows.append([CONFIG_HOST_ROOTPATH_KEY] + [self.host_rootpath])
        _csvrows.append([CONFIG_HOST_RESULTSDIR_KEY] + [self.host_resultspath])
        _csvrows.append([CONFIG_DEVICE_PATH_KEY] + [self.device_path])
        _csvrows.append([CONFIG_DEVICES_KEY] + [','.join(self.devices)])
        _csvrows.append([CONFIG_HOST_NAME_KEY] + [self.hostname])
        _csvrows.append([CONFIG_RUNS_KEY] + [self.iterations])
        _csvrows.append([CONFIG_MODEL_KEY + ":" + CONFIG_MODEL_NAME_SUBKEY] + [self.dnn_model.name])
        _csvrows.append([CONFIG_MODEL_KEY + ":" + CONFIG_MODEL_DLC_SUBKEY] + [self.dnn_model.dlc])
        _csvrows.append([CONFIG_MODEL_KEY + ":" + CONFIG_MODEL_INPUTLIST_SUBKEY] + [self.dnn_model.input_list_name])
        if 'RandomInput' in self._cfg_from_json[CONFIG_MODEL_KEY]:
            _csvrows.append([CONFIG_MODEL_KEY + ":" + CONFIG_MODEL_RANDOMINPUT_SUBKEY] + [self._cfg_from_json[CONFIG_MODEL_KEY][CONFIG_MODEL_RANDOMINPUT_SUBKEY]])
        else:
            for data in self._cfg_from_json[CONFIG_MODEL_KEY][CONFIG_MODEL_DATA_SUBKEY]:
                _csvrows.append([CONFIG_MODEL_KEY + ":" + CONFIG_MODEL_DATA_SUBKEY] + [data])
        _csvrows.append([CONFIG_MODEL_KEY + ":" + CONFIG_MODEL_INPUTS] + [','.join(self.dnn_model.input_layers)])
        _csvrows.append([CONFIG_RUNTIMES_KEY] + [','.join(self.return_valid_run_flavors())])
        _csvrows.append([CONFIG_ARCHITECTURES_KEY] + [','.join(self.architectures)])
        _csvrows.append([CONFIG_COMPILER_KEY] + [self.compiler])
        _csvrows.append([CONFIG_STL_LIBRARY_KEY] + [self.stl_library])
        _csvrows.append([CONFIG_MEASUREMENTS_KEY] + [','.join(self.measurements)])
        _csvrows.append([CONFIG_PERF_PROFILE_KEY] + [self.perfprofile])
        _csvrows.append(['Date'] + [datetime.datetime.now()])

        return _csvrows

    @property
    def name(self):
        return self._cfg_from_json[CONFIG_NAME_KEY]

    @property
    def host_rootpath(self):
        return self._cfg_from_json[CONFIG_HOST_ROOTPATH_KEY]

    @property
    def host_resultspath(self):
        return self._cfg_from_json[CONFIG_HOST_RESULTSDIR_KEY]

    @property
    def devices(self):
        return self._cfg_from_json[CONFIG_DEVICES_KEY]

    @property
    def hostname(self):
        return self._cfg_from_json[CONFIG_HOST_NAME_KEY]

    @property
    def device_path(self):
        return self._cfg_from_json[CONFIG_DEVICE_PATH_KEY]

    @property
    def iterations(self):
        return self._cfg_from_json[CONFIG_RUNS_KEY]

    @property
    def dnn_model(self):
        return self._dnnmodel

    @property
    def runtimes(self):
        return self._cfg_from_json[CONFIG_RUNTIMES_KEY]

    @property
    def userbuffer_mode(self):
        return self.userbuffer_mode

    def userbuffer_mode(self, value):
        self.userbuffer_mode = value

    @property
    def perfprofile(self):
        return self._cfg_from_json[CONFIG_PERF_PROFILE_KEY]

    def perfprofile(self, value):
        self.perfprofile = value

    @property
    def cpu_fallback(self):
        return self._cfg_from_json[CONFIG_CPU_FALLBACK_KEY]

    def cpu_fallback(self, value):
        self.cpu_fallback = value

    @property
    def architectures(self):
        return self._architectures

    @property
    def compiler(self):
        return self._compiler

    @property
    def platform(self):
        return self._platform_os

    def set_platform(self, platform_os):
        self._platform_os = platform_os

    @property
    def stl_library(self):
        return self._stl_library

    @property
    def measurements(self):
        return self._cfg_from_json[CONFIG_MEASUREMENTS_KEY]

    '''
       Support to add optional field "BufferTypes" : ["ub_float",...]
       when BufferTypes is not present, it runs with defaut behaviour i.e., adds all possible runtimes
       when "BufferTypes" : "", only adds all the runtimes given in "runtimes" field
       "BufferTypes" : valid user buffertype list, adds all the "runtimes" runtimes and their user buffertype variants
    '''

    def return_valid_run_flavors(self):
        flavors = []
        use_ub_float = True if self.buffertypes and USERBUFFER_FLOAT in self.buffertypes else False
        use_ub_tf8 = True if self.buffertypes and USERBUFFER_TF8 in self.buffertypes else False
        for rt in self.runtimes:
            if rt == RUNTIME_GPU and CONFIG_ARTIFACTS_COMPILER_KEY_TARGET_ANDROID_OSPACE in self._artifacts_config[CONFIG_ARTIFACTS_KEY] and not self.cpu_fallback:
                # TODO figure out a better way to decide whether GPU_s is supported
                flavors.append(RUNTIME_GPU_ONLY)
                if self.buffertypes == None or use_ub_float:
                    flavors.append(RUNTIME_GPU_ONLY + "_ub_float")
            flavors.append(rt) #adds any runtime given in "self.runtimes"
            if self.buffertypes == None or use_ub_float:
                flavors.append(rt + "_ub_float")
            if rt == RUNTIME_DSP and not self.cpu_fallback:
                if self.buffertypes == None or use_ub_tf8:
                    flavors.append(rt + "_ub_tf8")
            if rt == RUNTIME_CPU_MODE_FXP8:
                if self.buffertypes == None or use_ub_tf8:
                    flavors.append(rt + "_ub_tf8")
        return flavors

    @property
    def host_artifacts(self):
        _host_artifacts = {}
        for _compiler, _artifacts in self._artifacts_config[CONFIG_ARTIFACTS_KEY].items():
            if _compiler == CONFIG_ARTIFACTS_COMPILER_KEY_HOST:
                for _artifact_path in _artifacts:
                    if not os.path.isabs(_artifact_path):
                        _artifact_path = self.__default_artifact_path(_compiler, _artifact_path)
                    if os.path.exists(_artifact_path):
                        _base_name = os.path.basename(_artifact_path)
                        if _base_name == SNPE_DLC_INFO_EXE:
                            _host_artifacts[SNPE_DLC_INFO_EXE] = _artifact_path
                        elif _base_name == SNPE_DIAGVIEW_EXE:
                            _host_artifacts[SNPE_DIAGVIEW_EXE] = _artifact_path
                break
        return _host_artifacts

    @property
    def artifacts(self):
        _tmp = {}
        for _compiler, _artifacts in self._artifacts_config[CONFIG_ARTIFACTS_KEY].items():
            if _compiler == CONFIG_ARTIFACTS_COMPILER_KEY_HOST:
                continue
            _tmp[_compiler] = []
            _dev_bin_dir = os.path.join(self.device_path, ARTIFACT_DIR, _compiler, "bin")
            _dev_lib_dir = os.path.join(self.device_path, ARTIFACT_DIR, _compiler, "lib")
            for _artifact_path in _artifacts:
                if not os.path.isabs(_artifact_path):
                    _artifact_path = self.__default_artifact_path(_compiler, _artifact_path)
                if os.path.exists(_artifact_path):
                    _base_name = os.path.basename(_artifact_path)
                    if _base_name.endswith(".so") or _base_name.startswith("lib"):
                        _tmp[_compiler].append([_artifact_path, _dev_lib_dir])
                    else:
                        _tmp[_compiler].append([_artifact_path, _dev_bin_dir])
            if len(_tmp[_compiler]) == 0:
                del _tmp[_compiler]
        return _tmp

    @property
    def device_artifacts_bin(self):
        raise ConfigError('Deprecated call')

    @property
    def device_artifacts_lib(self):
        raise ConfigError('Deprecated call')

    def get_device_artifacts_bin(self, runtime):
        return os.path.join(self.__device_artifacts_helper(runtime), 'bin')

    def get_device_artifacts_lib(self, runtime):
        return os.path.join(self.__device_artifacts_helper(runtime), 'lib')

    def get_exe_name(self, runtime):
        if runtime == RUNTIME_GPU_ONLY and not self.cpu_fallback:
            return SNPE_BATCHRUN_GPU_S_EXE
        else:
            return SNPE_BATCHRUN_EXE

    def __device_artifacts_helper(self, runtime):
        '''
        Given a runtime (CPU, DSP, GPU, etc), return the artifacts folder

        Note that the logic is as simple as:
            1. There should be at least one artifacts folder.
            2. If you are asking for GPU_S, there needs to be 2
            3. GPU_S will return the one ending with "_s", all others will get the first one
        '''
        if runtime == RUNTIME_GPU_ONLY:
            anchor = SNPE_RUNTIME_GPU_S_LIB
        else:
            anchor = SNPE_RUNTIME_LIB
        for _compiler, _artifacts in self._artifacts_config[CONFIG_ARTIFACTS_KEY].items():
            if anchor in _artifacts:
                return os.path.join(self.device_path, ARTIFACT_DIR, _compiler)
        raise ConfigError('Unable to find device artifacts')

    def __str__(self):
        if 'RandomInput' in self._cfg_from_json[CONFIG_MODEL_KEY]:
            input_type = ('  %s:%s\n' % (CONFIG_MODEL_KEY + ":" + CONFIG_MODEL_RANDOMINPUT_SUBKEY, self._cfg_from_json[CONFIG_MODEL_KEY][CONFIG_MODEL_RANDOMINPUT_SUBKEY]))
        else:
            input_type = ('  %s:%s\n' % (CONFIG_MODEL_KEY + ":" + CONFIG_MODEL_DATA_SUBKEY, self._cfg_from_json[CONFIG_MODEL_KEY][CONFIG_MODEL_DATA_SUBKEY]))
        return ("\n--CONFIG--\n" +
                ('  %s:%s \n' % (CONFIG_NAME_KEY, self.name)) +
                ('  %s:%s \n' % (CONFIG_HOST_ROOTPATH_KEY, self.host_rootpath)) +
                ('  %s:%s \n' % (CONFIG_HOST_RESULTSDIR_KEY, self.host_resultspath)) +
                ('  %s:%s \n' % (CONFIG_DEVICES_KEY, self.devices)) +
                ('  %s:%s\n' % (CONFIG_DEVICE_PATH_KEY, self.device_path)) +
                ('  %s:%s\n' % (CONFIG_HOST_NAME_KEY, self.hostname)) +
                ('  %s:%s\n' % (CONFIG_RUNS_KEY, self.iterations)) +
                ('  %s:%s\n' % (CONFIG_MODEL_KEY + ":" + CONFIG_MODEL_NAME_SUBKEY , self._dnnmodel.name)) +
                ('  %s:%s\n' % (CONFIG_MODEL_KEY + ":" + CONFIG_MODEL_DLC_SUBKEY , self._dnnmodel.dlc)) +
                input_type +
                ('  %s:%s\n' % (CONFIG_MODEL_KEY + ":" + CONFIG_MODEL_INPUTS, self._dnnmodel.input_layers)) +
                ('  %s:%s\n' % (CONFIG_RUNTIMES_KEY, self.return_valid_run_flavors())) +
                ('  %s:%s\n' % (CONFIG_ARCHITECTURES_KEY, self.architectures)) +
                ('  %s:%s\n' % (CONFIG_COMPILER_KEY, self.compiler)) +
                ('  %s:%s\n' % (CONFIG_STL_LIBRARY_KEY, self.stl_library)) +
                ('  %s:%s\n' % (CONFIG_MEASUREMENTS_KEY, self.measurements)) +
                ('  %s:%s\n' % (CONFIG_PERF_PROFILE_KEY, self.perfprofile)) +
                ('  %s:%s\n' % (CONFIG_CPU_FALLBACK_KEY, self.cpu_fallback)) +
                ('  %s:%s\n' % (CONFIG_BUFFERTYPES_KEY, self.buffertypes)) +
                "--END CONFIG--\n")


class ConfigFactory:
    def __init__(self):
        pass

    @staticmethod
    def make_config(cfgfile, outputbasedir, devicelist, hostname, usedevicesconnected, deviceostype, logger, userbuffer_mode, perfprofile):
        _config_from_json = load_json(cfgfile, logger)
        if usedevicesconnected:
            devicelist = get_device_list(hostname, logger)
        elif not devicelist:
            devicelist = _config_from_json['Devices']

        dlc_path = _config_from_json[CONFIG_MODEL_KEY][CONFIG_MODEL_DLC_SUBKEY]
        if not os.path.exists(dlc_path):
            logger.error(dlc_path + " doesn't exist")
            return
        else:
            return Config(_config_from_json, outputbasedir, devicelist, hostname, deviceostype, logger, userbuffer_mode, perfprofile)
