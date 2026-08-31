"""On-device latency benchmark for papayaformer.onnx.

Copy papayaformer.onnx + this file to the target device (Android via Termux,
Raspberry Pi via SSH), install once:  pip install onnxruntime numpy
then:  python bench.py

Prints median / p95 latency at 1 and 4 threads, plus peak RAM.
"""
import time
import numpy as np
import onnxruntime as ort

X = np.random.randn(1, 3, 224, 224).astype(np.float32)

for threads in (1, 4):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession("papayaformer.onnx", so,
                                providers=["CPUExecutionProvider"])
    for _ in range(10):
        sess.run(None, {"x": X})
    t = []
    for _ in range(200):
        s = time.perf_counter()
        sess.run(None, {"x": X})
        t.append((time.perf_counter() - s) * 1000.0)
    t = np.array(t)
    print(f"{threads} thread : median {np.median(t):6.1f} ms | "
          f"p95 {np.percentile(t, 95):6.1f} ms | mean {t.mean():6.1f} ms")

try:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM"):
                print("peak RAM :", line.split()[1], "kB "
                      f"(~{int(line.split()[1]) / 1024:.0f} MB)")
except FileNotFoundError:
    print("peak RAM : (/proc not available on this OS)")
