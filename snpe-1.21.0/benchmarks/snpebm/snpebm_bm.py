from abc import ABCMeta, abstractproperty, abstractmethod
from snpebm_config_restrictions import *
from snpebm_parser import LogParserFactory
from collections import OrderedDict
import time
import os


class BenchmarkStat:
    def __init__(self, log_parser, stat_type):
        self._stats = []
        self._log_parser = log_parser
        self._type = stat_type
        return

    def __iter__(self):
        return self._stats.__iter__()

    @property
    def stats(self):
        return self._stats

    @property
    def type(self):
        return self._type

    def _process(self, input_dir, logger):
        data_frame = self._log_parser.parse(input_dir, logger)
        self._stats.append(data_frame)

    @property
    def average(self):
        avg_dict = OrderedDict()
        for stat in self._stats:
            for channel, _sum, _len, _max, _min, _runtime in stat:
                if channel in avg_dict:
                    avg_dict[channel][0] += _sum
                    avg_dict[channel][1] += _len
                else:
                    avg_dict[channel] = [_sum, _len]

        avgs = OrderedDict()
        for channel in avg_dict:
            avgs[channel] = avg_dict[channel][0] / avg_dict[channel][1]
        return avgs

    @property
    def max(self):
        max_dict = OrderedDict()
        for stat in self._stats:
            for channel, _sum, _len, _max, _min, _runtime in stat:
                if channel in max_dict:
                    max_dict[channel] = max(max_dict[channel], _max)
                else:
                    max_dict[channel] = _max
        return max_dict

    @property
    def min(self):
        min_dict = OrderedDict()
        for stat in self._stats:
            for channel, _sum, _len, _max, _min, _runtime in stat:
                if channel in min_dict:
                    min_dict[channel] = min(min_dict[channel], _min)
                else:
                    min_dict[channel] = _min
        return min_dict

    @property
    def runtime(self):
        runtime_dict = OrderedDict()
        for stat in self._stats:
            for channel, _sum, _len, _max, _min, _runtime in stat:
                if channel not in runtime_dict:
                    runtime_dict[channel] = _runtime
        return runtime_dict

class BenchmarkCommand:
    def __init__(self, cmd_str, shell=True):
        self._command = cmd_str
        self._is_shell = shell

    def command_str(self):
        return self._command

    def shell(self):
        return self._is_shell


class BenchmarkFactory:
    def __init__(self):
        pass

    @staticmethod
    def make_benchmarks(config):
        assert config, "config is required"
        assert config.measurement_types_are_valid(), "You asked for %s, but only these types of measurements are supported: %s"%(config.measurements,CONFIG_VALID_MEASURMENTS)
        host_result_dirs = {}
        for arch in config.architectures:
            if arch == ARCH_AARCH64 or arch == ARCH_ARM:
                if 'droid' not in host_result_dirs:
                    host_result_dirs['droid'] = \
                        SnapDnnCppDroidBenchmark.create_host_result_dir(config.host_resultspath)

        benchmarks = []
        for flavor in config.return_valid_run_flavors():
            for measurement in config.measurements:
                dev_bin_path = config.get_device_artifacts_bin(flavor)
                dev_lib_path = config.get_device_artifacts_lib(flavor)
                exe_name = config.get_exe_name(flavor)
                parser = LogParserFactory.make_parser(measurement, config)
                benchmarks.append(["{0}_{1}".format(str(flavor), measurement), SnapDnnCppDroidBenchmark(
                    dev_bin_path,
                    dev_lib_path,
                    exe_name,
                    config.dnn_model.device_rootdir,
                    os.path.basename(config.dnn_model.dlc),
                    config.dnn_model.input_list_name,
                    config.userbuffer_mode,
                    config.perfprofile,
                    config.cpu_fallback,
                    config.host_rootpath
                ).measurement(BenchmarkStat(parser, measurement))
                 .runtime(flavor)
                 .host_output_dir(host_result_dirs['droid'])
                 .name(flavor)])

        return benchmarks, host_result_dirs['droid']


class AbstractBenchmark(object):
    __metaclass__ = ABCMeta

    @abstractproperty
    def host_result_dir(self):
        return 'Derived class must implement this'

    @abstractproperty
    def exe_name(self):
        return 'Derived class must implement this'

    @abstractproperty
    def pre_commands(self):
        return 'Derived class must implement this'

    @abstractproperty
    def post_commands(self):
        return 'Derived class must implement this'

    @abstractproperty
    def commands(self):
        return 'Derived class must implement this'

    @abstractmethod
    def process_results(self):
        return 'Derived class must implement this'



class SnapDnnCppDroidBenchmark(AbstractBenchmark):
    @staticmethod
    def create_host_result_dir(host_output_dir):
        # Create results output dir, and a "latest_results" that links to it
        _now = time.localtime()[0:6]
        _host_output_datetime_dir = os.path.join(host_output_dir, SNPE_BENCH_OUTPUT_DIR_DATETIME_FMT % _now)
        os.makedirs(_host_output_datetime_dir)
        sim_link_path = os.path.join(host_output_dir, LATEST_RESULTS_LINK_NAME)
        if os.path.islink(sim_link_path):
            os.remove(sim_link_path)
        os.symlink(os.path.relpath(_host_output_datetime_dir,host_output_dir), sim_link_path)
        return _host_output_datetime_dir

    def __init__(self, exe_dir, dep_lib_dir, exe_name, model_dir, container_name, input_list_name, userbuffer_mode, perfprofile, cpu_fallback, host_rootpath):
        assert model_dir, "model dir is required"
        assert container_name, "container is required"
        assert input_list_name, "input_list is required"
        self._exe_dir = exe_dir
        self._model_dir = model_dir
        self._dep_lib_dir = dep_lib_dir
        self._exe_name = exe_name
        self._container = container_name
        self._input_list = input_list_name
        self._output_dir = 'output'
        self._host_output_dir = None
        self._host_result_dir = None
        self._debug = False
        self._runtime = RUNTIME_CPU
        self._rnn = False
        self._name = None
        self._run_number = 0
        self._measurement = None
        self.sh_path ='/system/bin/sh'
        self.userbuffer_mode = userbuffer_mode
        self.perfprofile = perfprofile
        self.cpu_fallback = cpu_fallback
        self.host_rootpath = host_rootpath
        return

    def host_output_dir(self, host_output_dir):
        self._host_output_dir = host_output_dir
        return self

    @property
    def host_result_dir(self):
        return self._host_result_dir

    def name(self, name):
        self._name = name
        return self

    def output_dir(self, output_dir):
        self._output_dir = output_dir
        return self

    def run_number(self, n):
        self._run_number = n

    def debug(self, debug):
        self._debug = debug
        return self

    def runtime(self, rt):
        self._runtime = rt
        return self

    def measurement(self, measurement):
        self._measurement = measurement
        return self

    def rnn(self, use_rnn):
        self._rnn = use_rnn
        return self

    @property
    def exe_name(self):
        return SNPE_BATCHRUN_EXE

    def __create_script(self):
        cmds = ['export LD_LIBRARY_PATH=' + self._dep_lib_dir + ':$LD_LIBRARY_PATH',
                'export ADSP_LIBRARY_PATH=\"' + self._dep_lib_dir + '/../../dsp/lib;/system/lib/rfsa/adsp;/usr/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp\"',
                'cd ' + self._model_dir,
                'rm -rf ' + self._output_dir]
        run_cmd = "{0} --container {1} --input_list {2} --output_dir {3}" \
            .format(os.path.join(self._exe_dir, self._exe_name), self._container, self._input_list, self._output_dir)
        if self._debug:
            run_cmd += " --debug"
        if RUNTIME_CPU_MODE_FXP8 in self._runtime:
            run_cmd += " --use_fxp_cpu"
        if RUNTIME_GPU in self._runtime or RUNTIME_GPU_ONLY in self._runtime:
            run_cmd += " --use_gpu"
        if RUNTIME_GPU_MODE_FP16 in self._runtime:
            run_cmd += " --gpu_mode " + GPU_MODE_FP16
        if RUNTIME_GPU_MODE_FP32 in self._runtime:
            run_cmd += " --gpu_mode " + GPU_MODE_DEFAULT
        if RUNTIME_DSP in self._runtime:
            run_cmd += " --use_dsp"
        if self._rnn:
            run_cmd += " --rnn_runtime"
        if self.userbuffer_mode == 'float' or "ub_float" in self._runtime:
            run_cmd += " --userbuffer_float"
        elif self.userbuffer_mode == 'tf8' or "ub_tf8" in self._runtime:
            run_cmd += " --userbuffer_tf8"
        if self.perfprofile:
            run_cmd += " --perf_profile " + self.perfprofile
        if self.cpu_fallback:
            run_cmd += " --enable_cpu_fallback"
        cmds.append(run_cmd)
        cmd_script_path = os.path.join(self.host_rootpath, SNPE_BENCH_SCRIPT)
        if os.path.isfile(cmd_script_path):
            os.remove(cmd_script_path)
        cmd_script = open(cmd_script_path, 'w')
        cmd_script.write('#!' + self.sh_path + '\n')
        for ln in cmds:
            cmd_script.write(ln + '\n')
        cmd_script.close()
        os.chmod(cmd_script_path, 0o555)
        return cmd_script_path

    @property
    def pre_commands(self):
        self._host_result_dir = os.path.join(
            self._host_output_dir, self._measurement.type, self._name, "Run" + str(self._run_number))
        os.makedirs(self._host_result_dir)
        cmd_script = self.__create_script()
        diag_rm_files = os.path.join(self._model_dir, SNPE_BENCH_DIAG_REMOVE)
        return [BenchmarkCommand("rm -f {0} ".format(diag_rm_files), shell = True),
                BenchmarkCommand("push {0} {1}".format(cmd_script, self._exe_dir), shell=False)]

    @property
    def commands(self):
        return [BenchmarkCommand("sh {0}".format(os.path.join(self._exe_dir, SNPE_BENCH_SCRIPT)))]

    @property
    def post_commands(self):
        if self._host_output_dir is None:
            return []
        device_output_dir = os.path.join(self._model_dir, self._output_dir)
        # now will also pull the script file used to generate the results
        return [BenchmarkCommand("chmod 777 {0}".format(device_output_dir)),
                BenchmarkCommand("pull {0} {1}".format(os.path.join(self._exe_dir,SNPE_BENCH_SCRIPT),
                                                       self._host_result_dir), shell=False),
                BenchmarkCommand("pull {0} {1}".format(os.path.join(device_output_dir,SNPE_BENCH_DIAG_OUTPUT_FILE),
                                                       self._host_result_dir), shell=False)]
    def get_snpe_version(self, config, logger):
        snpe_version_parser = LogParserFactory.make_parser(MEASURE_SNPE_VERSION, config)
        return snpe_version_parser.parse(self.host_result_dir, logger)

    def process_results(self, logger):
        assert os.path.isdir(self._host_result_dir), "ERROR: no host result directory"
        self._measurement._process(self._host_result_dir, logger)
        return

