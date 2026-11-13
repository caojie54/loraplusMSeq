import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.wait_for_gpus import select_gpus


class WaitForGpusTest(unittest.TestCase):
    def test_filters_and_required_count(self):
        gpus = [
            dict(index='0', uuid='GPU-a', name='A100', free=80000, utilization=90),
            dict(index='1', uuid='GPU-b', name='A100', free=71680, utilization=0),
            dict(index='2', uuid='GPU-c', name='H100', free=79000, utilization=0),
        ]
        self.assertEqual(select_gpus(gpus, 2, 71680, {'1', 'GPU-c'}), [gpus[2], gpus[1]])
        self.assertEqual(select_gpus(gpus, 2, 71680, name='A100', max_utilization=10), [])
        self.assertEqual(select_gpus(gpus, 1, 71680, name='A100', max_utilization=10), [gpus[1]])

    def test_retry_then_execute_once_with_environment_and_exit_code(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/wait_for_gpus.py'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / 'nvidia-smi'
            fake.write_text('''#!/bin/bash
count=0
[[ ! -f "$QUERY_COUNT" ]] || read -r count < "$QUERY_COUNT"
count=$((count + 1))
echo "$count" > "$QUERY_COUNT"
if (( count == 1 )); then echo 'temporary query error' >&2; exit 1; fi
echo '0, GPU-a, A100, 10000, 0'
echo '1, GPU-b, A100, 72000, 0'
if (( count > 2 )); then echo '2, GPU-c, A100, 80000, 0'; fi
''')
            fake.chmod(0o755)
            env = os.environ.copy()
            env.pop('CUDA_VISIBLE_DEVICES', None)
            env.pop('SLURM_JOB_GPUS', None)
            env.update(PATH=tmp + ':' + env['PATH'], QUERY_COUNT=str(root/'count'))
            target = root / 'task.json'
            command = [sys.executable, str(script), '--num-gpus', '2', '--min-free-gb', '70',
                       '--interval', '0.01', '--', sys.executable, '-c',
                       'import os,json,sys;json.dump(dict(os.environ),open(sys.argv[1],"w"));sys.exit(7)', str(target)]
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 7, result.stderr)
            self.assertEqual((root/'count').read_text().strip(), '3')
            actual = json.loads(target.read_text())
            self.assertEqual(actual['CUDA_VISIBLE_DEVICES'], '2,1')
            self.assertEqual(actual['REQUIRED_FREE_GB'], '0')
            self.assertEqual(result.stdout.count('; exec '), 1)


if __name__ == '__main__':
    unittest.main()
