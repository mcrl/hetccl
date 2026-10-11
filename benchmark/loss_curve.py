# Usage:
#  python loss_curve.py loss_curves/nvidia_8N_iter1000_gpt_wiki103_loss_curve.txt loss_curves/amd_8N_iter1000_gpt_wiki103_loss_curve.txt -l "NVIDIA" "AMD" -o nvidia_vs_amd_loss.png -t "GPT2 Training Loss Curve: NVIDIA vs AMD"
#  python loss_curve.py loss_curves/nvidia_8N_gpt2-small_lr5e-4_iter5000_wiki103.txt loss_curves/amd_8N_gpt2_small_lr5e-4_iter5000_wiki103.txt loss_curves/het_4A4N_gpt2-small_lr5e-4_iter5000_wiki103.txt -l "NCCL" "RCCL" "HetCCL" -o gpt2_loss_curve.png -t "GPT2 Training Loss Curve"

import argparse
import matplotlib.pyplot as plt
import numpy as np

def parse_loss_file(filepath):
    """Parse loss curve file with step and loss columns"""
    steps = []
    losses = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    steps.append(int(parts[0]))
                    losses.append(float(parts[1]))
                except ValueError:
                    continue
    
    return np.array(steps), np.array(losses)

def plot_loss_curves(files, labels=None, output='loss_curves.png', title='Training Loss Curves'):
    """Plot multiple loss curves on the same figure"""
    plt.figure(figsize=(12, 7))
    
    if labels is None:
        labels = [f"Curve {i+1}" for i in range(len(files))]
    
    # Different line styles and markers for better distinction
    styles = [
        {'linestyle': '-', 'marker': 'o', 'markersize': 3, 'alpha': 0.3},
        {'linestyle': '-', 'marker': 's', 'markersize': 3, 'alpha': 0.3},
        {'linestyle': '-', 'marker': '^', 'markersize': 3, 'alpha': 0.3},
        {'linestyle': '-', 'marker': 'd', 'markersize': 3, 'alpha': 0.3},
    ]
    
    for i, (filepath, label) in enumerate(zip(files, labels)):
        steps, losses = parse_loss_file(filepath)
        style = styles[i % len(styles)]
        plt.plot(steps, losses, linewidth=2.5, label=label, **style)
    
    plt.xlabel('Training steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"Loss curve plot saved to: {output}")
    
    # Also show statistics
    print("\n=== Loss Statistics ===")
    for filepath, label in zip(files, labels):
        steps, losses = parse_loss_file(filepath)
        if len(losses) > 0:
            print(f"{label}:")
            print(f"  Initial loss: {losses[0]:.4f}")
            print(f"  Final loss: {losses[-1]:.4f}")
            print(f"  Min loss: {losses.min():.4f} (step {steps[losses.argmin()]})")
            print(f"  Max loss: {losses.max():.4f} (step {steps[losses.argmax()]})")
            print(f"  Total steps: {len(steps)}")

def main():
    parser = argparse.ArgumentParser(description='Plot training loss curves from multiple files')
    parser.add_argument('files', nargs='+', help='Loss curve files to plot')
    parser.add_argument('-l', '--labels', nargs='+', help='Labels for each curve')
    parser.add_argument('-o', '--output', default='loss_curves.png', help='Output image file (default: loss_curves.png)')
    parser.add_argument('-t', '--title', default='Training Loss Curves', help='Plot title')
    
    args = parser.parse_args()
    
    if args.labels and len(args.labels) != len(args.files):
        print(f"Warning: Number of labels ({len(args.labels)}) doesn't match number of files ({len(args.files)})")
        print("Using default labels instead.")
        args.labels = None
    
    plot_loss_curves(args.files, args.labels, args.output, args.title)

if __name__ == '__main__':
    main()
