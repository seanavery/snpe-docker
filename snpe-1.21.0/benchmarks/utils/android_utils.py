import os
import re
import subprocess
from subprocess import check_output
from subprocess import CalledProcessError
from subprocess import STDOUT


UNKNOWN = 'unknown'
REGX_GET_PROP = re.compile('\[(.+)\]: \[(.+)\]')
getprop_list = ['ro.product.name',
                'ro.serialno',
                'ro.product.model',
                'ro.product.board',
                'ro.product.brand',
                'ro.product.device',
                'ro.product.manufacturer',
                'ro.product.cpu.abi',
                'ro.build.au_rev',
                'ro.build.description',
                'ro.build.version.sdk']
ADB_SHELL_CMD_SUCCESS = 'ADB_SHELL_CMD_SUCCESS'

class AdbShellCmdFailedException(Exception):
    def __str__(self):
        return('\nadb shell command Error: ' + repr(self) + '\n')

def get_device_list(hostname, logger):
    adb_cmd = 'adb devices'

    if hostname is not None:
        adb_cmd = 'adb -H ' + hostname + ' devices'
    device_list = []
    try:
        cmd_output = _execute_adbcmd_raw(adb_cmd, logger)
    except CalledProcessError as e:
        raise
    if cmd_output:
        regex_adb_devices = re.compile('^(.+)\s+device$')
        for line in cmd_output.split('\n'):
            m = regex_adb_devices.search(line)
            if m and m.group(1):
                device_list.append(m.group(1))
    return device_list


def get_device_info(device, logger, fatal=True):
    _info = {}
    getprop_cmd = "shell getprop | grep \"{0}\"".format('\|'.join(getprop_list))
    try:
        props = execute_adbcmd(device, getprop_cmd, logger)
    except CalledProcessError as e:
        if fatal != True:
            logger.warning('Non fatal get prop call failure, is the target os not Android?')
            return []
        raise
    if props:
        for line in props.split('\n'):
            m = REGX_GET_PROP.search(line)
            if m:
                _info[m.group(1)] = m.group(2)
    dev_info = []
    for prop_key in getprop_list:
        if not prop_key in _info:
            dev_info.append([prop_key, UNKNOWN])
        else:
            dev_info.append([prop_key, _info[prop_key]])
    return dev_info


def generate_adbcmd(device, cmd, shell=False):
    """
    Returns an adb command string that can be executed on the target
    """
    host = device.host_name
    dev_serial = device.comm_id
    if not shell:
        cmd_str = 'adb -H %s -s %s %s' % (host, dev_serial, cmd)
    else:
        cmd_str = 'adb -H %s -s %s shell \"%s && echo %s\"' % (host, dev_serial, cmd, ADB_SHELL_CMD_SUCCESS)
    return cmd_str

def _execute_adbcmd_raw(cmd_str, logger, shell=False, suppress_warning=False):
    cmd_handle = ''
    try:
        logger.debug('Executing {%s}' % cmd_str)
        p = subprocess.Popen(cmd_str, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        cmd_out, cmd_err = p.communicate()
        cmd_out = cmd_out.decode()
        cmd_err = cmd_err.decode()
        returncode = p.returncode
        # check shell executed successfully on target
        if shell and (ADB_SHELL_CMD_SUCCESS not in cmd_out or returncode is not 0):
            # Between adb version 1.0.32 to 1.0.39, there is a change that now propagates
            # failed adb command properly (e.g. return code, also failure would show up
            # in stderr rather than stdout.  In order to work with both, we will need to
            # pass back both stdout and stderr to be processed by the caller, who may be
            # looking for particular substring (e.g. check for file existence).  Note this
            # is porcelain
            if not suppress_warning:
                logger.error('%s failed with stderr of: %s'%(cmd_str, cmd_out + cmd_err))
            raise AdbShellCmdFailedException(cmd_out + cmd_err)
        logger.debug('Command Output: \n"%s"' % cmd_out)
        return cmd_out
    except AdbShellCmdFailedException as e:
        if not suppress_warning:
            logger.warning('adb shell command failed to execute:\n\t%s' % repr(e))
        raise

def execute_adbcmd(device, cmd, logger, shell=False, suppress_warning=False):
    """
    Runs a BLOCKING adb command on target and raises exception
    when an error is encountered
    """
    cmd_str = generate_adbcmd(device, cmd, shell)
    return _execute_adbcmd_raw(cmd_str, logger, shell, suppress_warning)

def test_device_access(device, logger):
    """
    Tests access to android target connected to host device
    """
    access_cmd = 'get-state'
    try:
        cmd_handle = execute_adbcmd(device, access_cmd, logger)
        cmd_success = 'device' in cmd_handle.strip()
        if not cmd_success:
            logger.debug('Output from the get-state command %s' % cmd_handle)
            assert cmd_success, 'Could not run simple command on %s' % device
    except Exception as e:
        raise AdbShellCmdFailedException('Could not run simple command on device %s' % device.comm_id)


def check_file_exists(device, file_path, logger, suppress_warning=False):
    """
    Returns 'True' if the file exists on the target
    """
    shell_cmd = "ls %s" % file_path
    try:
        execute_adbcmd(device, shell_cmd, logger, shell=True, suppress_warning=suppress_warning)
    except AdbShellCmdFailedException as e:
        if 'No such file or directory' in repr(e):
            return False
        else: # throw exception at caller: some other issue occurred
            raise
    else:
        # ls returning 0 means file/directory exists
        return True


def push_file(device, host_src_path, device_dest_dir, logger, silent=False):
    """
    A summary of the function

    Args:
        host_src_path: file/dir to be pushed
        device_dest_dir: destination folder on device
        device: DriodDevice object
        logger: logger object
        silent: To pipe the stdout to null

    Returns:
        True if the push is successful

    Raises:
        RuntimeError when the source_path does not exist
    """
    if not os.path.exists(host_src_path):
        logger.error('Path %s does not exist' % host_src_path)
        raise('%s is not a file or directory', RuntimeError)
    else:
        src_name = os.path.basename(host_src_path)
        device_dest_path = os.path.join(device_dest_dir, src_name)
        logger.debug('Pushing %s to %s on %s' % (host_src_path, device_dest_path, device))
        push_cmd = 'push %s %s' % (host_src_path, device_dest_path)
        if silent:
            push_cmd += ' >/dev/null 2>&1'
        cmd_handle = execute_adbcmd(device, push_cmd, logger)
        logger.debug('Pushed %s to %s' % (host_src_path, device))
        return cmd_handle


def pull_file(device, device_src_path, host_dest_dir, logger, silent=False):
    """
    A summary of the function

    Args:
        device_src_path: file/dir to be pulled
        host_dest_dir: destination folder path
        device: DriodDevice object
        logger: logger object
        silent: To pipe the stdout to null

    Returns:
        True if the pull is successful
    """
    src_name = os.path.basename(device_src_path)
    host_dest_path = os.path.join(host_dest_dir, src_name)

    logger.debug('Pulling %s to %s from %s' % (device_src_path, host_dest_path, device))

    pull_cmd = 'pull %s %s' % (device_src_path, host_dest_path)

    if silent:
        pull_cmd += ' >/dev/null 2>&1'

    cmd_handle = execute_adbcmd(device, pull_cmd, logger)

    # catching errors
    no_error = cmd_handle.find('error') == -1
    file_found = cmd_handle.find('does not exist') == -1
    try:
        assert no_error, 'adb pull command threw an error on %s' % device
        assert file_found, 'Could not find the file %s on %s' % (device_src_path, device)
    except Exception as e:
        logger.error('Could not run pull command on %s' % device, exc_info=True)
        logger.error(repr(e))
        raise
    else:
        logger.info('Pulled %s to %s' % (device_src_path, host_dest_dir))
        return cmd_handle, host_dest_path
