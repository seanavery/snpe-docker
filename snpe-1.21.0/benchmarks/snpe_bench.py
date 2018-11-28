import logging
import sys
import os
from time import sleep
sys.path.insert(0, os.path.join(os.getcwd(), 'utils'))
sys.path.insert(0, os.path.join(os.getcwd(), 'snpebm'))
from snpebm_bm import *
from snpebm_config import *
from snpebm_device import *
from snpebm_parser import *
from snpebm_csvwriter import *
from snpebm_md5 import perform_md5_check


def __get_logger(debug,device_id=None):

    if device_id:
        log_prefix = SNPE_BENCH_NAME+'_'+device_id
    else:
        log_prefix = SNPE_BENCH_NAME
    logger = logging.getLogger(log_prefix)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    if not len(logger.handlers):
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(logging.Formatter(SNPE_BENCH_LOG_FORMAT.format(log_prefix)))
        logger.addHandler(stdout_handler)
    return logger



def _find_shell_binary_on_target(device, logger):
    sh_path = '/system/bin/sh'
    if check_file_exists(device, sh_path, logger, suppress_warning=True) == False:
        sh_cmd = 'shell which sh'
        sh_path = ''
        try:
            sh_path = execute_adbcmd(device, sh_cmd, logger)
        except CalledProcessError as e:
            sh_path = ''
        if sh_path == '' or "not found" in sh_path:
            logger.error('Could not find md5 checksum binary on device.')
            sh_path = ''
    return sh_path.rstrip()

def snpe_bench(program_name,args_list, device_msm_os_dict=None):
    try:
        args_parser = ArgsParser(program_name,args_list)
        if args_parser.device_id_override:
            logger = __get_logger(args_parser.debug_enabled,args_parser.device_id_override[0])
        else:
            logger = __get_logger(args_parser.debug_enabled)

        logger.info("Running {0} with {1}".format(SNPE_BENCH_NAME, args_parser.args))

        config = ConfigFactory.make_config(args_parser.config_file_path, args_parser.output_basedir_override, args_parser.device_id_override, args_parser.host_name,
                                           args_parser.run_on_all_connected_devices_override, args_parser.device_os_type_override, logger,
                                           args_parser.userbuffer_mode, args_parser.perfprofile)
        if config is None:
            return
        logger.info(config)

        # Dictionary is {"cpu_arm_all_Memory":ZdlSnapDnnCppDroidBenchmark object}
        benchmarks, results_dir = BenchmarkFactory.make_benchmarks(config)

        # Now loop through all the devices and run the benchmarks on them
        for device_id in config.devices:
            device = DeviceFactory.make_device(device_id, config)
            test_device_access(device, logger)
            device_info = get_device_info(device, logger, fatal=((args_parser.device_os_type_override != 'le' and args_parser.device_os_type_override != 'le64')))
            logger.debug("Perform md5 checksum on %s"%device_id)
            perform_md5_check(device,[item for sublist in config.artifacts.values() for item in sublist] + config.dnn_model.artifacts, logger)
            logger.debug("Artifacts on %s passed checksum"%device_id)
            sh_path = _find_shell_binary_on_target(device, logger)

            benchmarks_ran = []
            # Run each benchmark on device, and pull results
            for run_flavor_measure, bm in benchmarks:
                logger.info(run_flavor_measure)
                bm.sh_path = sh_path
                # running iterations of the same runtime.  Two possible failure cases:
                # 1. Say GPU runtime is not available
                # 2. Transient failure
                # For now, for either of those cases, we will mark the whole runtime
                # as bad, so I break out of for loop as soon as a failure is detected
                for i in range(1, config.iterations + 1):
                    logger.info("Run " + str(i))
                    bm.run_number(i)
                    try:
                        device.execute(bm.pre_commands, logger)
                        device.start_measurement(bm, logger)
                        #Sleep to let things cool off
                        if args_parser.sleep != 0:
                            logger.debug("Sleeping: " + str(args_parser.sleep))
                            sleep(args_parser.sleep)

                        device.execute(bm.commands, logger)
                        device.stop_measurement(bm, logger)
                        device.execute(bm.post_commands, logger)
                    except AdbShellCmdFailedException as e:
                        logger.warning('Failed to perform benchmark for %s.' % run_flavor_measure)
                        break
                    finally:
                        device.stop_measurement(bm, logger)

                    bm.process_results(logger)
                else:  # Ran through iterations without failing
                    benchmarks_ran.append((run_flavor_measure, bm))

            if len(benchmarks_ran) == 0:
                logger.error('None of the selected benchmarks ran, therefore no results reported')
                sys.exit(ERRNUM_NOBENCHMARKRAN_ERROR)
            else:
                if(device_msm_os_dict != None):
                    chipset = ('Chipset' , device_msm_os_dict[device_id][1])
                    OS = ()
                    if(device_msm_os_dict[device_id][2] == ''):
                        OS = ('OS', device_msm_os_dict[device_id][3])
                    else:
                        OS = ('OS', device_msm_os_dict[device_id][2])
                    device_info.append(chipset)
                    device_info.append(OS)
                csv_writer = CsvWriterFactory.make_csv_writer(benchmarks_ran, config, device_info, args_parser.sleep, logger)
                csv_writer.write(os.path.join(results_dir, "benchmark_stats_{0}.csv".format(config.name)), logger)

    except ConfigError as ce:
        print(ce)
        sys.exit(ERRNUM_CONFIG_ERROR)
    except AdbShellCmdFailedException as ae:
        print(ae)
        sys.exit(ERRNUM_ADBSHELLCMDEXCEPTION_ERROR)
    except Exception as e:
        print(e)
        sys.exit(ERRNUM_GENERALEXCEPTION_ERROR)

if __name__ == "__main__":
    snpe_bench(sys.argv[0],sys.argv[1:])
