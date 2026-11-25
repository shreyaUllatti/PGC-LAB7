"""
Hardware CUDA-Event Timeline Profiler for Experiment 7
------------------------------------------------------
When CUPTI kernel trace export is restricted on Windows (WDDM / non-admin),
torch.profiler still records operator CUDA times in key_averages(), while
Chrome trace JSON omits raw `kernel` / `gpu_memcpy` categories.

This script instruments the exact same baseline and optimized workloads with
hardware `torch.cuda.Event(enable_timing=True)` timestamps (microsecond
resolution on the GPU) and exports Chrome-trace JSON files compatible with
`analyze_trace.py` and `ui.perfetto.dev`.
"""
import json
import math
import torch
from gpu_workload import make_weights, make_host_batches


def profile_baseline_events(host, w, reps=3, nbatches=10):
    t_origin = torch.cuda.Event(enable_timing=True)
    records = []
    torch.cuda.synchronize()
    t_origin.record()

    for i in range(nbatches):
        e_h2d_s, e_h2d_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e_h2d_s.record()
        x = host[i % len(host)].cuda()
        e_h2d_e.record()
        records.append(("gpu_memcpy", "Memcpy HtoD (Pageable -> Device)", e_h2d_s, e_h2d_e, 1))

        y = x
        for _ in range(reps):
            e1, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e1.record()
            y = y @ w
            e2.record()
            records.append(("kernel", "ampere_sgemm_128x64_nn (cuBLAS GEMM)", e1, e2, 2))

            e1, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e1.record()
            y = torch.relu(y)
            e2.record()
            records.append(("kernel", "elementwise_kernel (relu / clamp_min)", e1, e2, 2))

            e1, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e1.record()
            y = y.mul_(0.5)
            e2.record()
            records.append(("kernel", "elementwise_kernel (mul_)", e1, e2, 2))

            e1, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e1.record()
            y = y.add_(1.0)
            e2.record()
            records.append(("kernel", "elementwise_kernel (add_)", e1, e2, 2))

        e1, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e1.record()
        r = y.sum(dim=1)
        e2.record()
        records.append(("kernel", "reduce_kernel (sum dim=1)", e1, e2, 2))

        e_d2h_s, e_d2h_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e_d2h_s.record()
        _ = r.cpu()
        e_d2h_e.record()
        records.append(("gpu_memcpy", "Memcpy DtoH (Device -> Pageable)", e_d2h_s, e_d2h_e, 1))

    torch.cuda.synchronize()
    return to_chrome_trace(t_origin, records)


def profile_optimized_events(host, w, reps=3, nbatches=10):
    rows, cols = host[0].shape
    copy_s = torch.cuda.Stream()
    comp_s = torch.cuda.Stream()
    buf = [torch.empty(rows, cols, device="cuda") for _ in range(2)]
    copied = [torch.cuda.Event() for _ in range(2)]
    freed = [torch.cuda.Event() for _ in range(2)]
    out_host = torch.empty(nbatches, rows).pin_memory()

    t_origin = torch.cuda.Event(enable_timing=True)
    records = []

    torch.cuda.synchronize()
    t_origin.record(copy_s)
    for e in freed:
        e.record(comp_s)

    def issue_copy(i):
        k = i % 2
        copy_s.wait_event(freed[k])
        with torch.cuda.stream(copy_s):
            e1, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e1.record(copy_s)
            buf[k].copy_(host[i % len(host)], non_blocking=True)
            e2.record(copy_s)
            copied[k].record(copy_s)
            records.append(("gpu_memcpy", "Memcpy HtoD (Pinned -> Device)", e1, e2, 1))

    issue_copy(0)
    for i in range(nbatches):
        k = i % 2
        if i + 1 < nbatches:
            issue_copy(i + 1)
        comp_s.wait_event(copied[k])
        with torch.cuda.stream(comp_s):
            y = buf[k]
            for _ in range(reps):
                e1, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e1.record(comp_s)
                y = y @ w
                e2.record(comp_s)
                records.append(("kernel", "ampere_sgemm_128x64_nn (cuBLAS GEMM)", e1, e2, 2))

                e1, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e1.record(comp_s)
                y = torch.relu(y)
                e2.record(comp_s)
                records.append(("kernel", "elementwise_kernel (relu / clamp_min)", e1, e2, 2))

                e1, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e1.record(comp_s)
                y = y.mul_(0.5)
                e2.record(comp_s)
                records.append(("kernel", "elementwise_kernel (mul_)", e1, e2, 2))

                e1, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e1.record(comp_s)
                y = y.add_(1.0)
                e2.record(comp_s)
                records.append(("kernel", "elementwise_kernel (add_)", e1, e2, 2))

            e1, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e1.record(comp_s)
            r = y.sum(dim=1)
            e2.record(comp_s)
            records.append(("kernel", "reduce_kernel (sum dim=1)", e1, e2, 2))
            freed[k].record(comp_s)

            e1, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e1.record(comp_s)
            out_host[i].copy_(r, non_blocking=True)
            e2.record(comp_s)
            records.append(("gpu_memcpy", "Memcpy DtoH (Device -> Pinned)", e1, e2, 2))

    torch.cuda.synchronize()
    return to_chrome_trace(t_origin, records)


def to_chrome_trace(t_origin, records):
    events = []
    for cat, name, e_s, e_e, tid in records:
        ts_us = t_origin.elapsed_time(e_s) * 1000.0
        dur_us = e_s.elapsed_time(e_e) * 1000.0
        if dur_us < 0:
            dur_us = 0.0
        events.append({
            "ph": "X",
            "cat": cat,
            "name": name,
            "pid": 0,
            "tid": tid,
            "ts": round(ts_us, 2),
            "dur": round(dur_us, 2),
        })
    return {"traceEvents": events}


def main():
    rows, cols, reps, nbatches = 16384, 1024, 3, 10
    w = make_weights(cols)

    # Warm-up
    host_b = make_host_batches(rows, cols, pinned=False)
    profile_baseline_events(host_b, w, reps, 3)
    tb = profile_baseline_events(host_b, w, reps, nbatches)
    with open("results/trace_baseline_events.json", "w") as f:
        json.dump(tb, f)

    host_o = make_host_batches(rows, cols, pinned=True)
    profile_optimized_events(host_o, w, reps, 3)
    to = profile_optimized_events(host_o, w, reps, nbatches)
    with open("results/trace_optimized_events.json", "w") as f:
        json.dump(to, f)

    print("Saved results/trace_baseline_events.json and results/trace_optimized_events.json")


if __name__ == "__main__":
    main()
