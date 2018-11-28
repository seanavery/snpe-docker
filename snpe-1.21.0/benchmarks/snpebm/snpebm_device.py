from abc import ABCMeta, abstractproperty, abstractmethod
from multiprocessing import Process
from android_utils import *
from fs_utils import *
from snpebm_constants import *
import os
import time
import re


REGX_SPACES = re.compile('[\s]+')
ONE_HOUR_IN_SEC = 1 * 60 * 60.0


class DeviceFactory():
    @staticmethod
    def make_device(device_id, config):
        assert device_id, "device id is required"
        assert config, "config is required"
        return BenchmarkDevice(device_id, device_id, config.device_path, config.platform, config.host_rootpath, config.hostname)

class AbstractBenchmarkDevice(object):
    __metaclass__ = ABCMeta

    @abstractproperty
    def comm_id(self):
        """
        Communication ID for the device
        """
        return 'Derived class must implement this'

    @abstractproperty
    def device_type(self):
        """
        Type of device
        """
        return 'Derived class must implement this'

    @abstractproperty
    def host_name(self):
        """
        Remote Host location
        """
        return 'Derived class must implement this'

    @abstractproperty
    def device_name(self):
        """
        Device name
        """
        return 'Derived class must implement this'

    @abstractproperty
    def device_root_dir(self):
        """
        Root directory on device to run benchmarks
        """
        return 'Derived class must implement this'

    @abstractproperty
    def host_output_dir(self):
        """
        Output directory on the host
        """
        return 'Derived class must implement this'

    @abstractmethod
    def setup_artifacts(self, logger):
        """
        Sets up all the benchmark artifacts on the device e.g.
        benchmarks, dependent libraries, etc.

        Args:
            logger: logger to be used

        Returns: None

        """
        return 'Derived class must implement this'

    @abstractmethod
    def setup_dnnmodel(self, model, logger):
        """
        Sets up model related files on the device e.g. dlc, input data etc.

        Args:
            model: dnn model
            logger: logger to be used

        Returns: None

        """
        return 'Derived class must implement this'

    @abstractmethod
    def execute(self, benchmark, logger):
        """

        Executes a benchmark
        Args:
            benchmark: the benchmark to be executed
            logger: logger to be used

        Returns:

        """
        return 'Derived class must implement this'

    @abstractmethod
    def start_measurement(self, benchmark, logger):
        """
        Start measurement for the benchmark
        Args:
            benchmark: benchmark to be measured
            logger: logger to be used

        Returns:

        """
        return 'Derived class must implement this'

    @abstractmethod
    def stop_measurement(self, benchmark, logger):
        """
        Stop measurement for the benchmark
        Args:
            benchmark: benchmark to be measured
            logger: logger to be used

        Returns:

        """
        return 'Derived class must implement this'


class BenchmarkDevice(AbstractBenchmarkDevice):
    def __init__(self, device_name, serial_no, device_root_dir, platform, host_output_dir, host_name='localhost'):
        assert device_root_dir, "device root directory is required"
        self._device_name = device_name
        self._comm_id = serial_no
        self._device_root_dir = device_root_dir
        self._host_output_dir = host_output_dir
        self._host_name = host_name
        self._mem_proc = None
        self._power_proc = None
        self._platform = platform

        if(self._platform == PLATFORM_OS_ANDROID):
            self._device_type = DEVICE_TYPE_ARM_ANDROID
        elif (self._platform == PLATFORM_OS_LINUX):
            self._device_type = DEVICE_TYPE_ARM_LINUX
        else:
            raise Exception("device: Invalid platform !!!", platform)

        return

    def __str__(self):
        return (('[Device Name:%s ' % self._device_name) +
                ('Device ID:%s ' % self._comm_id) +
                ('HOST NAME:%s ' % self.host_name) +
                ('Device DIR:%s]' % self._device_root_dir))

    @property
    def device_name(self):
        return self._device_name

    @property
    def host_name(self):
        return self._host_name

    @property
    def comm_id(self):
        return self._comm_id

    @property
    def device_type(self):
        return self._device_type

    @property
    def device_root_dir(self):
        return self._device_root_dir

    @property
    def host_output_dir(self):
        return self._host_output_dir

    def __create_dirs(self, dir_paths, logger):
        for dir_path in dir_paths:
            if check_file_exists(self, dir_path, logger):
                logger.debug('Deleting remote directory: %s' % dir_path)
                del_dir = 'rm -rf %s' % dir_path
                execute_adbcmd(self, del_dir, logger, shell=True)
                logger.info('Deleting existing directory {%s}' % dir_path)
            create_dir = 'mkdir -p %s' % dir_path
            execute_adbcmd(self, create_dir, logger, shell=True)
            logger.info('Created directory {%s}' % dir_path)
            change_perm = 'chmod 777 %s' % dir_path
            execute_adbcmd(self, change_perm, logger, shell=True)
            logger.info('Changed permissions for the directory {%s}' % dir_path)
        return

    def __copy_host_to_device(self, host_dir, device_dir, logger):
        host_files, host_dirs = recursive_dir_scan(host_dir)
        dev_dirs = []
        for host_dir_path in host_dirs:
            dev_dir_path = host_dir_path.replace(host_dir, device_dir, 1)
            dev_dirs.append(dev_dir_path)
        self.__create_dirs(dev_dirs, logger)
        for host_file_path in host_files:
            dev_file_path = os.path.dirname(host_file_path.replace(host_dir, device_dir, 1))
            push_file(self, host_file_path, dev_file_path, logger)
        return

    def __mem_log_file(self):
        return os.path.join(self._device_root_dir, MEM_LOG_FILE_NAME)

    def __capture_mem_droid(self, exe_name, logger):
        time_out = ONE_HOUR_IN_SEC
        t0 = time.time()
        ps_name = exe_name

        # Find the Process ID
        ps_pid = None
        while time_out > (time.time() - t0):
            version_output = execute_adbcmd(self, "getprop ro.build.version.release", logger, shell=True)
            android_version = version_output.strip().split()[0]
            if android_version >= "8.0.0":
                ps_output = execute_adbcmd(self, "ps -A | grep {0}".format(ps_name), logger, shell=True)
            else:
                ps_output = execute_adbcmd(self, "ps | grep {0}".format(ps_name), logger, shell=True)
            if ps_output:
                ps_pid = REGX_SPACES.split(ps_output.strip())[1]
                logger.debug(ps_output)
                logger.debug("Found PID ({0}) of the Process".format(ps_pid))
                break
            if ps_pid is not None:
                break

        assert ps_pid, "ERROR: Could not find the Process ID of {0}".format(exe_name)

        num_of_samples = 0
        mem_log_file = self.__mem_log_file()
        logger.debug("Capturing memory usage of {0} with PID {1}".format(exe_name, ps_pid))
        logger.debug("Time required to determine the PID:{0}".format((time.time() - t0)))
        while time_out > (time.time() - t0):
            if num_of_samples == 0:
                logger.debug("Memory Log Capture available at: {0}".format(mem_log_file))
                create_or_append = ">"
            else:
                create_or_append = "| cat >>"
            execute_adbcmd(self, "dumpsys meminfo {0} {2} {1}".format(ps_pid, mem_log_file, create_or_append), logger, shell=True)
            num_of_samples += 1
        return

    def __capture_mem_le(self, exe_name, logger):
        time_out = ONE_HOUR_IN_SEC
        t0 = time.time()
        ps_name = exe_name

        # Find the Process ID
        ps_pid = None
        execute_adbcmd(self, "uname", logger, shell=True)
        while time_out > (time.time() - t0):
            output_pid = execute_adbcmd(self, "ps -A | grep {0}".format(ps_name), logger, shell=True)
            if output_pid:
                ps_pid = REGX_SPACES.split(output_pid.strip())[0]
                logger.debug(output_pid)
                logger.debug("Found PID ({0}) of the Process".format(ps_pid))
                break

        assert ps_pid, "Could not find the Process ID of {0}".format(exe_name)

        num_of_samples = 0
        mem_log_file = self.__mem_log_file()
        logger.debug("Capturing memory usage of {0} with PID {1}".format(exe_name, ps_pid))
        logger.debug("Time required to determine the PID:{0}".format((time.time() - t0)))
        while time_out > (time.time() - t0):
            if num_of_samples == 0:
                logger.debug("Memory Log Capture available at: {0}".format(mem_log_file))
                create_or_append = ">"
            else:
                create_or_append = "| cat >>"
            execute_adbcmd(self, "cat /proc/{0}/smaps | cat >> {1}".format(ps_pid,mem_log_file),logger,shell=True)
            execute_adbcmd(self, "echo ==== {2} {1}".format(ps_pid, mem_log_file, create_or_append), logger, shell=True)
            num_of_samples += 1
        return

    def setup_artifacts(self, artifacts_dict, logger):
        for _compiler, _host_to_dev_artifacts in artifacts_dict.items():
            for _host_file, _dev_dir in _host_to_dev_artifacts:
                if not check_file_exists(self, _dev_dir, logger):
                    self.__create_dirs([_dev_dir], logger)
                push_file(self, _host_file, _dev_dir, logger)

    def setup_dnnmodel(self, model, logger):
        for _host_path, _dev_dir in model.artifacts:
            if not check_file_exists(self, _dev_dir, logger):
                self.__create_dirs([_dev_dir], logger)
            if os.path.isdir(_host_path):
                self.__copy_host_to_device(_host_path, _dev_dir, logger)
            else:
                push_file(self, _host_path, _dev_dir, logger)
        return

    def execute(self, commands, logger):
        for cmd in commands:
            execute_adbcmd(self, cmd.command_str(), logger, shell=cmd.shell())
        return

    def start_measurement(self, benchmark, logger):
        if benchmark._measurement.type == MEASURE_MEM:
            if self._mem_proc is None:
                logger.info("starting memory capture in a parallel process")
                if(self._platform == PLATFORM_OS_ANDROID):
                    logger.info("Android platform")
                    self._mem_proc = Process(target=self.__capture_mem_droid, args=(benchmark.exe_name, logger))
                    self._mem_proc.start()
                elif(self._platform == PLATFORM_OS_LINUX):
                    logger.info("Linux Embedded")
                    self._mem_proc = Process(target=self.__capture_mem_le, args=(benchmark.exe_name, logger))
                    self._mem_proc.start()
                else:
                    raise Exception("start_measurement: Invalid platform !!!", self.platform)
            else:
                logger.info("memory capture is already started")
        return

    def stop_measurement(self, benchmark, logger):
        if benchmark._measurement.type == MEASURE_MEM:
            if self._mem_proc is not None:
                self._mem_proc.terminate()
                self._mem_proc = None
                logger.info("memory capture is terminated")
                execute_adbcmd(self, "pull {0} {1}".format(self.__mem_log_file(), benchmark.host_result_dir), logger)
        return
