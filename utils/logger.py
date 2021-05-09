import logging
import os
import pathlib
import platform
import shlex
import struct
import subprocess
import warnings
from datetime import datetime
from typing import Union

from colorlog import ColoredFormatter

# Suppressing DeprecationWarnings
warnings.filterwarnings("ignore")

step_num = 1
step_symbol = '-'


def _get_terminal_size_windows():
    try:
        from ctypes import windll, create_string_buffer
        # stdin handle is -10
        # stdout handle is -11
        # stderr handle is -12
        h = windll.kernel32.GetStdHandle(-12)
        csbi = create_string_buffer(22)
        res = windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)
        if res:
            (bufx, bufy, curx, cury, wattr,
             left, top, right, bottom,
             maxx, maxy) = struct.unpack("hhhhHhhhhhh", csbi.raw)
            sizex = right - left + 1
            sizey = bottom - top + 1
            return sizex, sizey
    except:
        pass


def _get_terminal_size_tput():
    # get terminal width
    # src: http://stackoverflow.com/questions/263890/how-do-i-find-the-width-height-of-a-terminal-window
    try:
        cols = int(subprocess.check_call(shlex.split('tput cols')))
        rows = int(subprocess.check_call(shlex.split('tput lines')))
        return (cols, rows)
    except:
        pass


def _get_terminal_size_linux():
    def ioctl_GWINSZ(fd):
        try:
            import fcntl
            import termios
            cr = struct.unpack('hh',
                               fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))
            return cr
        except:
            pass

    cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
    if not cr:
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            cr = ioctl_GWINSZ(fd)
            os.close(fd)
        except:
            pass
    if not cr:
        try:
            cr = (os.environ['LINES'], os.environ['COLUMNS'])
        except:
            return None
    return int(cr[1]), int(cr[0])


def get_terminal_size():
    current_os = platform.system()
    tuple_xy = None
    if current_os == 'Windows':
        tuple_xy = _get_terminal_size_windows()
        if tuple_xy is None:
            tuple_xy = _get_terminal_size_tput()
            # needed for window's python in cygwin's xterm!
    if current_os in ['Linux', 'Darwin'] or current_os.startswith('CYGWIN'):
        tuple_xy = _get_terminal_size_linux()
    if tuple_xy is None:
        tuple_xy = (80, 25)  # default value
    return tuple_xy


num_symbols = get_terminal_size()[0] - 1


def create_logger(name: str = None, logging_mode: str = 'INFO', file_logging_mode: str = 'DEBUG',
                  log_to_file: bool = False, log_location: Union[str, pathlib.Path] = None,
                  log_name: str = 'log') -> logging.Logger:
    console_formatter = ColoredFormatter(
        '%(blue)s%(asctime)s%(reset)s - %(name)s - %(log_color)s%(levelname)s%(reset)s - %(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger_inst = logging.getLogger(name=name)
    logger_inst.setLevel('DEBUG')
    if log_to_file is True:
        if not log_location and log_to_file:
            log_location = pathlib.Path(__file__).parent.absolute()
        log_name = pathlib.Path().joinpath(log_location,
                                           log_name + '_' + str(datetime.now().strftime("%Y-%m-%d_%H_%M_%S")) + '.log')
        fh = logging.FileHandler(log_name)
        fh.setLevel(file_logging_mode)
        fh.setFormatter(file_formatter)
        logger_inst.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging_mode)
    ch.setFormatter(console_formatter)
    logger_inst.addHandler(ch)
    return logger_inst


def step(msg: str, logger: logging) -> None:
    global step_num, num_symbols
    step_symbol = '-'
    logger.info(str("\n\n" + step_symbol * num_symbols) +
                "\nStep {}: {}\n".format(step_num, msg) +
                str(step_symbol * num_symbols))
    step_num += 1


if __name__ == "__main__":
    logger = create_logger(logging_mode='DEBUG', file_logging_mode='DEBUG', log_to_file=False)

    step('a', logger)
    logger.log(logging.INFO, 'b')
    step('b', logger)
    logger.info('c')
    logger.warning('d')
    logger.error('e')
    logger.debug('f')


    class MyClass:
        _log = logging.getLogger('MyClass')

        def __init__(self):
            self._log.info('Init')


    my_class = MyClass()
