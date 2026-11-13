"""Wait for GPUs with enough free memory, then exec a command once (no GPU reservation)."""

import argparse
import csv
import datetime
import math
import os
import shlex
import subprocess
import sys
import time


def log(message):
    print(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def query_gpus():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=30,
    )
    gpus = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) != 5:
            continue
        index, uuid, name, free, utilization = (value.strip() for value in row)
        if not index.isdigit() or not free.isdigit():
            continue
        gpus.append(dict(index=index, uuid=uuid, name=name, free=int(free),
                         utilization=int(utilization) if utilization.isdigit() else None))
    return gpus


def select_gpus(gpus, count, free_mib, candidates=None, name=None, max_utilization=None):
    eligible = []
    for gpu in gpus:
        if candidates is not None and not ({gpu['index'], gpu['uuid']} & candidates):
            continue
        if gpu['free'] < free_mib or (name and name.lower() not in gpu['name'].lower()):
            continue
        if max_utilization is not None and (
            gpu['utilization'] is None or gpu['utilization'] > max_utilization
        ):
            continue
        eligible.append(gpu)
    eligible.sort(key=lambda gpu: (-gpu['free'], int(gpu['index'])))
    return eligible[:count] if len(eligible) >= count else []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--min-free-gb', type=float, default=70, help='Minimum free GiB per GPU')
    parser.add_argument('--interval', type=float, default=60, help='Polling interval in seconds')
    parser.add_argument('--gpu-ids', help='Candidate nvidia-smi indices or full UUIDs, comma-separated; otherwise inherit CUDA_VISIBLE_DEVICES/SLURM_JOB_GPUS')
    parser.add_argument('--name-contains', help='Optional GPU model-name filter, e.g. A100')
    parser.add_argument('--max-utilization', type=int, help='Optional maximum GPU utilization percent')
    parser.add_argument('--device-format', choices=['index', 'uuid'], default='index',
                        help='CUDA_VISIBLE_DEVICES values; uuid avoids CUDA/index ordering differences')
    parser.add_argument('command', nargs=argparse.REMAINDER, help='Command after --, e.g. -- bash task.sh')
    args = parser.parse_args()
    if args.num_gpus < 1 or not math.isfinite(args.min_free_gb) or args.min_free_gb < 0:
        parser.error('num-gpus must be positive and min-free-gb non-negative')
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error('interval must be positive')
    if args.max_utilization is not None and not 0 <= args.max_utilization <= 100:
        parser.error('max-utilization must be between 0 and 100')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('provide a task command after --')
    candidate_text = args.gpu_ids
    if candidate_text is None:
        candidate_text = os.environ.get('CUDA_VISIBLE_DEVICES', os.environ.get('SLURM_JOB_GPUS'))
    candidates = None if candidate_text is None else {s.strip() for s in candidate_text.split(',') if s.strip()}
    if candidates is not None and (len(candidates) < args.num_gpus or '-1' in candidates):
        parser.error('candidate list must contain at least num-gpus visible devices')
    log(f"Waiting for {args.num_gpus} GPU(s), each >= {args.min_free_gb:g} GiB free; interval={args.interval:g}s; candidates={candidate_text if candidate_text is not None else 'all'}")
    while True:
        try:
            gpus = query_gpus()
        except FileNotFoundError:
            parser.exit(2, 'nvidia-smi not found\n')
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            log(f"GPU query failed: {getattr(exc, 'stderr', '') or str(exc)}; retrying")
        else:
            selected = select_gpus(gpus, args.num_gpus, args.min_free_gb * 1024,
                                   candidates, args.name_contains, args.max_utilization)
            if selected:
                visible = ','.join(gpu[args.device_format] for gpu in selected)
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = visible
                env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
                # Existing MedReason task scripts should keep this selection.
                env['REQUIRED_FREE_GB'] = '0'
                log('Selected ' + ', '.join(f"GPU {g['index']} ({g['name']}, {g['free']} MiB free)" for g in selected))
                log(f"CUDA_VISIBLE_DEVICES={visible}; exec {shlex.join(command)}")
                os.execvpe(command[0], command, env)
            log('Not enough qualifying GPUs; waiting')
        time.sleep(args.interval)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
