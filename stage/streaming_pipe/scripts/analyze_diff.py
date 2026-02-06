#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze the differences between 30ms and 10ms versions in detail
"""

import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STREAMING_FBANK_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
BUILD_DIR = os.path.join(STREAMING_FBANK_DIR, "build")

def main():
    print("=" * 60)
    print("Detailed Difference Analysis")
    print("=" * 60)
    print()
    
    # Load both files
    file_30ms = os.path.join(BUILD_DIR, "streaming_fbank_30ms.npy")
    file_10ms = os.path.join(BUILD_DIR, "streaming_fbank_10ms.npy")
    
    fbank_30ms = np.load(file_30ms)
    fbank_10ms = np.load(file_10ms)
    
    print(f"30ms shape: {fbank_30ms.shape}")
    print(f"10ms shape: {fbank_10ms.shape}")
    print()
    
    # Calculate differences
    diff = np.abs(fbank_30ms - fbank_10ms)
    
    # Find frames with largest differences
    frame_max_diffs = np.max(diff[0], axis=1)
    frame_mean_diffs = np.mean(diff[0], axis=1)
    
    print("Per-frame statistics:")
    print(f"  Max diff across all frames: {np.max(frame_max_diffs):.6f}")
    print(f"  Mean diff across all frames: {np.mean(frame_mean_diffs):.6f}")
    print()
    
    # Find top 10 frames with largest differences
    top_frames = np.argsort(frame_max_diffs)[-10:][::-1]
    
    print("Top 10 frames with largest differences:")
    for i, frame_idx in enumerate(top_frames):
        max_diff = frame_max_diffs[frame_idx]
        mean_diff = frame_mean_diffs[frame_idx]
        max_dim = np.argmax(diff[0, frame_idx])
        print(f"  {i+1}. Frame {frame_idx}: max={max_diff:.6f} (dim {max_dim}), mean={mean_diff:.6f}")
    print()
    
    # Analyze the pattern
    print("Difference pattern analysis:")
    
    # Check if differences are concentrated in certain dimensions
    dim_max_diffs = np.max(diff[0], axis=0)
    dim_mean_diffs = np.mean(diff[0], axis=0)
    
    top_dims = np.argsort(dim_max_diffs)[-10:][::-1]
    print("Top 10 dimensions with largest differences:")
    for i, dim_idx in enumerate(top_dims):
        max_diff = dim_max_diffs[dim_idx]
        mean_diff = dim_mean_diffs[dim_idx]
        print(f"  {i+1}. Dim {dim_idx}: max={max_diff:.6f}, mean={mean_diff:.6f}")
    print()
    
    # Check if differences grow over time
    print("Difference over time:")
    num_segments = 10
    segment_size = len(frame_max_diffs) // num_segments
    for i in range(num_segments):
        start = i * segment_size
        end = (i + 1) * segment_size if i < num_segments - 1 else len(frame_max_diffs)
        segment_max = np.max(frame_max_diffs[start:end])
        segment_mean = np.mean(frame_mean_diffs[start:end])
        print(f"  Frames {start:3d}-{end:3d}: max={segment_max:.6f}, mean={segment_mean:.6f}")
    print()
    
    # Show the worst frame in detail
    worst_frame = top_frames[0]
    print(f"Detailed view of worst frame (frame {worst_frame}):")
    print(f"  30ms version (first 10 values): {fbank_30ms[0, worst_frame, :10]}")
    print(f"  10ms version (first 10 values): {fbank_10ms[0, worst_frame, :10]}")
    print(f"  Difference   (first 10 values): {diff[0, worst_frame, :10]}")
    print()
    
    # Check if the difference is systematic (e.g., scaling or offset)
    print("Checking for systematic differences:")
    
    # Correlation
    flat_30ms = fbank_30ms[0].flatten()
    flat_10ms = fbank_10ms[0].flatten()
    correlation = np.corrcoef(flat_30ms, flat_10ms)[0, 1]
    print(f"  Correlation: {correlation:.8f}")
    
    # Ratio
    ratio = flat_30ms / (flat_10ms + 1e-10)
    print(f"  Mean ratio (30ms/10ms): {np.mean(ratio):.6f}")
    print(f"  Std ratio: {np.std(ratio):.6f}")
    
    print()
    print("=" * 60)

if __name__ == "__main__":
    main()
