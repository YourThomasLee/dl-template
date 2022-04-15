import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import commands
import datetime
import functools
import time

def time_counter(func):
    """compute  execution time of function"""
    count_time = None
    if sys.version[0] == "3":
        count_time = time.perf_counter
    elif sys.version[0] == "2":
        count_time = time.clock
    start = count_time()
    @functools.wraps(func)
    def _wrapper(*args, **kvargs):
        return func(*args, **kvargs)
    end = count_time()
    print("the function %s excecution time is %s" % (func.__name__, str(datetime.timedelta(seconds = end - start))))
    return _wrapper

def run_shell(shell_str):
    """execute shell commands"""
    logger.debug("shell_str: %s" % shell_str)
    status, output = commands.getstatusoutput(shell_str)
    if status != 0:
        logger.error(output)
        return 1
    else:
        logger.debug("achieve excuting shell: %s" % shell_str)
        # logger.debug("shell output: %s" % output)
        return 0

def get_chunks(file_size, chunk_size = 0x20000):
    """
       get array of chunk size
       chunk_size: 131072 bytes = 128 kb, default max ssl buffer size
    """
    chunk_start = 0
    while chunk_start + chunk_size < file_size:
        yield(chunk_start, chunk_size)
        chunk_start += chunk_size
    final_chunk_size = file_size - chunk_start
    yield(chunk_start, final_chunk_size)

def read_file(file_path):
    """efficient method to read a (large) file into program
    @input: a file uri
    @return: a list generator of lines in file.
    """
    with open(file_path, "r") as fin:
        file_size = os.path.getsize(file_path)
        progress = 0
        prior_content = ""
        for chunk_start, chunk_size in get_chunks(file_size):
            file_chunk = fin.read(chunk_size)
            progress += len(file_chunk)
            #logger.debug('{0} of {1} bytes read ({2}%)'.format(
            #    progress, file_size, int(progress / file_size * 100))
            #    )
            lines = file_chunk.split("\n")
            n_lines = len(lines)
            for idx, line in enumerate(lines):
                # condition 1: last part is "", it is perfect
                # condition 2: last part is "xsffsghsgsh", it may be incomplete
                if idx == 0:
                    line_ret = (prior_content + line).strip()
                elif idx + 1 == n_lines:
                    if len(line.strip()) == 0:
                        prior_content = ""
                    else:
                        prior_content = line
                    continue
                else:
                    line_ret = line.strip()
                
                # return a line
                if len(line_ret) == 0 or (len(line_ret)>0 and line_ret[0]) == "#":
                    continue
                else:
                    yield line_ret


def ensure_dir(dirname):
    """make sure a directory existed """
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    """read json"""
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    """write json"""
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
