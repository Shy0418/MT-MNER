"""
Efficiency Metrics Module for MT-MNER
Computes: parameter counts, FLOPs, inference latency, training time, GPU memory
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, Tuple, Optional, List
from collections import defaultdict


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {'total': total, 'trainable': trainable, 'frozen': frozen}


def count_flops_hook(model, input_ids, attention_mask, pixel_values_vit, pixel_values_res):
    hooks = []
    flop_counts = defaultdict(float)

    def _hook_linear(module, inputs, output, name):
        inp = inputs[0]
        if isinstance(inp, torch.Tensor) and inp.numel() > 0:
            b = inp.shape[0]
            flops = 2 * b * module.in_features * module.out_features
            flop_counts[name] += flops

    def _hook_conv2d(module, inputs, output, name):
        inp = inputs[0]
        if isinstance(inp, torch.Tensor) and inp.numel() > 0:
            b = output.shape[0]
            oh, ow = output.shape[2], output.shape[3]
            k_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
            flops = 2 * b * oh * ow * k_ops * module.out_channels
            flop_counts[name] += flops

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            h = module.register_forward_hook(lambda m, i, o, n=name: _hook_linear(m, i, o, n))
            hooks.append(h)
        elif isinstance(module, nn.Conv2d):
            h = module.register_forward_hook(lambda m, i, o, n=name: _hook_conv2d(m, i, o, n))
            hooks.append(h)

    training = model.training
    model.eval()
    with torch.no_grad():
        _ = model(input_ids, attention_mask, pixel_values_vit, pixel_values_res, labels=None)
    model.train(training)

    for h in hooks:
        h.remove()

    total_flops = sum(flop_counts.values())
    return {'total_flops': float(total_flops), 'total_gflops': float(total_flops / 1e9)}


def count_flops_thop(model, input_ids, attention_mask, pixel_values_vit, pixel_values_res):
    try:
        from thop import profile as thop_profile

        training = model.training
        model.eval()

        class _Wrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, ids, am, pv_vit, pv_res):
                return self.m(ids, am, pv_vit, pv_res, labels=None)

        wrapper = _Wrapper(model)
        macs, params = thop_profile(wrapper, inputs=(input_ids, attention_mask, pixel_values_vit, pixel_values_res), verbose=False)
        model.train(training)

        return {'total_flops': float(macs * 2), 'total_gflops': float(macs * 2 / 1e9)}
    except ImportError:
        return count_flops_hook(model, input_ids, attention_mask, pixel_values_vit, pixel_values_res)


def measure_inference_latency(model, dataloader, device, num_samples=200, warmup_samples=50):
    model.eval()
    per_sample_times = []
    samples_processed = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values_vit = batch['pixel_values_vit'].to(device)
            pixel_values_res = batch['pixel_values_res'].to(device)
            bs = input_ids.size(0)

            if samples_processed < warmup_samples:
                _ = model.predict(input_ids, attention_mask, pixel_values_vit, pixel_values_res)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                samples_processed += bs
                if samples_processed >= warmup_samples:
                    continue
                continue

            if device.type == 'cuda':
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model.predict(input_ids, attention_mask, pixel_values_vit, pixel_values_res)
                end.record()
                torch.cuda.synchronize()
                elapsed = start.elapsed_time(end)
            else:
                t0 = time.perf_counter()
                _ = model.predict(input_ids, attention_mask, pixel_values_vit, pixel_values_res)
                elapsed = (time.perf_counter() - t0) * 1000.0

            per_sample_times.append(elapsed / bs)
            samples_processed += bs

            if samples_processed - warmup_samples >= num_samples:
                break

    if not per_sample_times:
        return {'avg_latency_ms': 0.0, 'avg_latency_s': 0.0}

    avg_ms = float(np.mean(per_sample_times))
    return {'avg_latency_ms': avg_ms, 'avg_latency_s': avg_ms / 1000.0}


def get_gpu_memory_usage(device):
    if device.type == 'cuda' and torch.cuda.is_available():
        allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        return {'max_allocated_mb': allocated, 'max_reserved_mb': reserved}
    return {'max_allocated_mb': 0.0, 'max_reserved_mb': 0.0}


def reset_gpu_memory_stats(device):
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def measure_all_efficiency(model, dataloader, device, training_time_s, sample_batch=None):
    params = count_parameters(model)

    if sample_batch is None:
        sample_batch = next(iter(dataloader))
    input_ids = sample_batch['input_ids'][:1].to(device)
    attention_mask = sample_batch['attention_mask'][:1].to(device)
    pixel_values_vit = sample_batch['pixel_values_vit'][:1].to(device)
    pixel_values_res = sample_batch['pixel_values_res'][:1].to(device)

    flops = count_flops_thop(model, input_ids, attention_mask, pixel_values_vit, pixel_values_res)
    latency = measure_inference_latency(model, dataloader, device)
    gpu_memory = get_gpu_memory_usage(device)

    return {
        'params': params,
        'flops': flops,
        'latency': latency,
        'training_time_s': training_time_s,
        'gpu_memory': gpu_memory,
    }


def print_efficiency_summary(metrics):
    params = metrics['params']
    flops = metrics['flops']
    latency = metrics['latency']
    training_time = metrics['training_time_s']
    gpu_memory = metrics['gpu_memory']

    sep = '=' * 65
    sub = chr(9472) * 45

    print()
    print(sep)
    print('  EFFICIENCY METRICS SUMMARY')
    print(sep)

    print()
    print('  Model Parameters')
    print('  ' + sub)
    print('    Total parameters      : {:>12,}'.format(params['total']))
    print('    Trainable parameters   : {:>12,}'.format(params['trainable']))
    print('    Frozen parameters      : {:>12,}'.format(params['frozen']))

    print()
    print('  FLOPs (Floating Point Operations)')
    print('  ' + sub)
    print('    Per forward pass       : {:>10.2f} GFLOPs'.format(flops['total_gflops']))
    print('    (raw)                  : {:>12,.0f} FLOPs'.format(flops['total_flops']))

    print()
    print('  Inference Latency (average per sample)')
    print('  ' + sub)
    print('    Per sample             : {:>10.2f} ms'.format(latency['avg_latency_ms']))
    print('                            {:>10.4f} s'.format(latency['avg_latency_s']))

    print()
    print('  Total Training Time')
    print('  ' + sub)
    h = int(training_time // 3600)
    m = int((training_time % 3600) // 60)
    s = training_time % 60
    print('    Wall-clock time        : {:>3d}h {:>02d}m {:>04.1f}s'.format(h, m, s))
    print('    (seconds)              : {:>10.1f} s'.format(training_time))

    print()
    print('  GPU Memory Usage (peak)')
    print('  ' + sub)
    print('    Max allocated          : {:>10.2f} MiB'.format(gpu_memory['max_allocated_mb']))
    print('    Max reserved (cached)  : {:>10.2f} MiB'.format(gpu_memory['max_reserved_mb']))

    print()
    print(sep)
    print()
