#!/usr/bin/env python
"""Inspect Core ML's estimated device assignment for the shipped backbone.

This is a compilation plan, not a per-request hardware trace. Use Instruments or
powermetrics while inference runs to confirm actual ANE activity.
"""
import argparse
from collections import Counter
from pathlib import Path

import coremltools as ct

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlpackage", default=str(
        ROOT / "hf-model/kev-qwen3-0.6b-hidden.mlpackage"))
    ap.add_argument("--compute-units", default="CPU_AND_NE",
                    choices=("CPU_AND_NE", "CPU_ONLY", "CPU_AND_GPU", "ALL"))
    args = ap.parse_args()
    units = getattr(ct.ComputeUnit, args.compute_units)
    model = ct.models.MLModel(args.mlpackage, compute_units=units)
    if model.__proxy__ is None:
        raise SystemExit("Core ML model did not compile; inspect the warning above")
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        model.get_compiled_model_path(), compute_units=units)
    operations = plan.model_structure.program.functions["main"].block.operations
    devices = Counter()
    estimated_cost = Counter()
    unestimated = Counter()
    by_type = Counter()
    for operation in operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
        device = (type(usage.preferred_compute_device).__name__
                  if usage is not None else "unreported")
        devices[device] += 1
        cost = plan.get_estimated_cost_for_mlprogram_operation(operation)
        if cost is None:
            unestimated[device] += 1
        else:
            estimated_cost[device] += cost.weight
        by_type[(operation.operator_name, device)] += 1
    print(f"model: {args.mlpackage}")
    print(f"allowed compute units: {args.compute_units}")
    print(f"operations: {len(operations)}")
    print("estimated preferred device:")
    for device, count in devices.most_common():
        print(f"  {device}: {count}")
    print("estimated model workload by preferred device:")
    for device, weight in estimated_cost.most_common():
        print(f"  {device}: {weight:.1%}")
    print(f"operations without cost estimate: {sum(unestimated.values())}")
    print("main compute operations:")
    for (op, device), count in by_type.most_common():
        if any(name in op for name in ("conv", "matmul", "softmax", "layer_norm")):
            print(f"  {op}: {count} on {device}")
    print("This plan estimates device placement; it does not measure runtime ANE use.")


if __name__ == "__main__":
    main()
