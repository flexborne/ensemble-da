"""The machine and software a benchmark ran on (no host names, user names or file paths)."""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

import ensemble_da


def _cpu_model():
    try:
        if sys.platform == 'darwin':
            return subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True,
                                  text=True, check=True).stdout.strip()
        if sys.platform.startswith('linux'):
            for line in Path('/proc/cpuinfo').read_text().splitlines():
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or 'unknown'


def _blas():
    info = {}
    for name, mod in (('numpy', np), ('scipy', scipy)):
        try:
            deps = mod.show_config(mode='dicts').get('Build Dependencies', {})
            info[name] = {k: {f: deps[k].get(f) for f in ('name', 'version')} for k in ('blas', 'lapack') if k in deps}
        except Exception as exc:                  # releases without show_config(mode='dicts')
            info[name] = f'unavailable: {exc}'
    try:
        import threadpoolctl
        info['threadpoolctl'] = [{k: v for k, v in d.items() if k not in ('filepath', 'prefix')}
                                 for d in threadpoolctl.threadpool_info()]
    except ImportError:
        info['threadpoolctl'] = 'not installed'
    return info


def _git_commit():
    """HEAD commit, suffixed '-dirty' when tracked files have uncommitted changes."""
    here = Path(__file__).resolve().parent
    try:
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=here, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=here,
                               capture_output=True, text=True, check=True).stdout.strip()
        return head + ('-dirty' if dirty else '')
    except (OSError, subprocess.SubprocessError):
        return None


def environment():
    return {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'python': platform.python_version(), 'implementation': platform.python_implementation(),
        'os': f'{platform.system()} {platform.release()}', 'machine': platform.machine(),
        'cpu': _cpu_model(), 'logical_cpus': os.cpu_count(),
        'numpy': np.__version__, 'scipy': scipy.__version__, 'ensemble_da': ensemble_da.__version__,
        'blas': _blas(), 'thread_env': {k: v for k, v in os.environ.items() if k.endswith('_THREADS')},
        'git_commit': _git_commit(),
    }
