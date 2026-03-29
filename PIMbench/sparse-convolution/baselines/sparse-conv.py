# PyTorch code to run sparse convolution on CPU or GPU with CUDA support
# Implements the three-stage sparse convolution pipeline from Zhang et al. (2025):
#   1. Valid Sub-Convolution Detection
#   2. Sparse im2col
#   3. GEMM Integration
import argparse
import torch
import time


def generate_sparse_input(batch_size, channels, height, width, sparsity, device):
    # Generate a sparse input tensor simulating event-based camera data.
    # Non-zero pixels represent brightness-change events. The sparsity parameter controls the fraction of elements that are zero (e.g. 0.95 means 95% zeros).
    dense = torch.randn(batch_size, channels, height, width, device=device)
    mask = (torch.rand(batch_size, channels, height, width, device=device) >= sparsity).float()
    return dense * mask


def valid_subconv_detection(input_tensor, kernel_height, kernel_width, stride):
    # Stage 1: Identify valid sub-convolution positions using active pixel locations (Valid_pix), following Zhang et al. Section 3.2
    # Returns a list of (row, col) tensors per batch element and the output spatial dimensions.
    batch_size, channels, height, width = input_tensor.shape
    out_h = (height - kernel_height) // stride + 1
    out_w = (width - kernel_width) // stride + 1
 
    valid_positions = []
    for b in range(batch_size):
        # Collect Valid_pix: spatial positions where any channel is non-zero
        active_mask = input_tensor[b].abs().sum(dim=0) > 0  # (H, W)
        active_coords = torch.nonzero(active_mask, as_tuple=False)  # (N, 2)
 
        if active_coords.shape[0] == 0:
            valid_positions.append(
                torch.zeros(0, 2, dtype=torch.long, device=input_tensor.device)
            )
            continue
 
        # For each active pixel (r, c), compute the range of output window
        # positions (oh, ow) whose receptive field includes that pixel.
        #   oh_min = ceil(max(0, r - kH + 1) / stride)
        #   oh_max = floor(min(r, (out_h - 1) * stride) / stride)
        # Same logic for the width axis.
        r = active_coords[:, 0]
        c = active_coords[:, 1]
 
        oh_min = torch.clamp((r - kernel_height + 1).float().div(stride).ceil().long(), min=0)
        oh_max = torch.clamp(r // stride, max=out_h - 1)
        ow_min = torch.clamp((c - kernel_width + 1).float().div(stride).ceil().long(), min=0)
        ow_max = torch.clamp(c // stride, max=out_w - 1)
 
        # Expand each active pixel into all output positions it maps to and
        # collect the union via a flat boolean mask.
        valid_mask = torch.zeros(out_h, out_w, dtype=torch.bool,
                                 device=input_tensor.device)
        for idx in range(active_coords.shape[0]):
            if oh_min[idx] <= oh_max[idx] and ow_min[idx] <= ow_max[idx]:
                valid_mask[oh_min[idx]:oh_max[idx] + 1,
                           ow_min[idx]:ow_max[idx] + 1] = True
 
        positions = torch.nonzero(valid_mask, as_tuple=False)
        valid_positions.append(positions)
 
    return valid_positions, out_h, out_w


def sparse_im2col(input_tensor, valid_positions, kernel_height, kernel_width,
                  stride):
    #Stage 2: Gather only the valid image patches into a compact dense matrix.

    # For each valid window position the corresponding (channels * kH * kW) patch is extracted and stacked into columns, producing a matrix of shape (batch, channels * kH * kW, num_valid_positions) per batch element.
    batch_size, channels, height, width = input_tensor.shape
    patch_size = channels * kernel_height * kernel_width

    gathered_columns = []
    for b in range(batch_size):
        positions = valid_positions[b]
        num_valid = positions.shape[0]
        if num_valid == 0:
            gathered_columns.append(
                torch.zeros(patch_size, 0, device=input_tensor.device)
            )
            continue

        rows = positions[:, 0] * stride
        cols = positions[:, 1] * stride

        # Extract patches for all valid positions at once
        patches = torch.zeros(num_valid, patch_size, device=input_tensor.device)
        for idx in range(num_valid):
            r, c = rows[idx].item(), cols[idx].item()
            patch = input_tensor[b, :, r:r + kernel_height, c:c + kernel_width]
            patches[idx] = patch.reshape(-1)

        gathered_columns.append(patches.t())

    return gathered_columns


def sparse_gemm(gathered_columns, kernel_weights, valid_positions, out_h, out_w,
                batch_size, device):
    # Stage 3: Perform GEMM between the kernel weight matrix and the gathered sparse columns, then scatter results back into the full output tensor.
    # kernel_weights shape: (out_channels, in_channels * kH * kW)
    # gathered_columns[b] shape: (in_channels * kH * kW, num_valid)
    
    out_channels = kernel_weights.shape[0]
    output = torch.zeros(batch_size, out_channels, out_h, out_w, device=device)

    for b in range(batch_size):
        cols = gathered_columns[b]
        if cols.shape[1] == 0:
            continue

        # Core GEMM: (out_channels, patch_size) x (patch_size, num_valid)
        result = kernel_weights @ cols  # (out_channels, num_valid)

        positions = valid_positions[b]
        for idx in range(positions.shape[0]):
            r, c = positions[idx, 0].item(), positions[idx, 1].item()
            output[b, :, r, c] = result[:, idx]

    return output


def sparse_convolution(input_tensor, kernel_weights, kernel_height,
                       kernel_width, stride):
    # Run the full three-stage sparse convolution pipeline.
    valid_positions, out_h, out_w = valid_subconv_detection(
        input_tensor, kernel_height, kernel_width, stride
    )
    gathered_columns = sparse_im2col(
        input_tensor, valid_positions, kernel_height, kernel_width, stride
    )
    output = sparse_gemm(
        gathered_columns, kernel_weights, valid_positions, out_h, out_w,
        input_tensor.shape[0], input_tensor.device
    )
    return output, valid_positions


def dense_convolution_reference(input_tensor, conv_layer, device):
    # Standard dense convolution for correctness comparison.
    input_tensor = input_tensor.to(device)
    conv_layer = conv_layer.to(device)
    return conv_layer(input_tensor)


def main(args):
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Input: {args.batch_size}x{args.input_depth}x{args.input_height}x{args.input_width}")
    print(f"[INFO] Kernel: {args.kernel_depth}x{args.input_depth}x{args.kernel_height}x{args.kernel_width}")
    print(f"[INFO] Sparsity: {args.sparsity:.1%} zeros")

    # Generate sparse input simulating event camera data
    input_tensor = generate_sparse_input(
        args.batch_size, args.input_depth, args.input_height, args.input_width,
        args.sparsity, device
    )

    actual_sparsity = (input_tensor == 0).float().mean().item()
    print(f"[INFO] Actual sparsity: {actual_sparsity:.1%} zeros")

    # Kernel weights reshaped for GEMM: (out_channels, in_channels * kH * kW)
    kernel_weights = torch.randn(
        args.kernel_depth,
        args.input_depth * args.kernel_height * args.kernel_width,
        device=device,
    )

    # --- Sparse convolution timing ---
    start_time = time.time()
    sparse_output, valid_positions = sparse_convolution(
        input_tensor, kernel_weights, args.kernel_height, args.kernel_width,
        args.stride,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    sparse_time = time.time() - start_time

    total_positions = sparse_output.shape[2] * sparse_output.shape[3]
    valid_count = sum(p.shape[0] for p in valid_positions)
    print(f"[INFO] Valid sub-convolutions: {valid_count} / "
          f"{total_positions * args.batch_size} total "
          f"({valid_count / (total_positions * args.batch_size):.1%})")
    print(f"[INFO] Sparse convolution time: {sparse_time * 1000:.6f} ms")

    # --- Dense convolution timing for comparison ---
    conv_layer = torch.nn.Conv2d(
        in_channels=args.input_depth,
        out_channels=args.kernel_depth,
        kernel_size=(args.kernel_height, args.kernel_width),
        stride=args.stride,
        padding=args.padding,
        bias=False,
    ).to(device)

    start_time = time.time()
    dense_output = conv_layer(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dense_time = time.time() - start_time

    print(f"[INFO] Dense convolution time:  {dense_time * 1000:.6f} ms")
    print(f"[INFO] Speedup (dense / sparse): {dense_time / sparse_time:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sparse convolution baseline for event-based camera data"
    )

    parser.add_argument("-b", "--batch_size", type=int, default=1,
                        help="Batch size for the input")
    parser.add_argument("-d", "--input_depth", type=int, default=3,
                        help="Depth (channels) of the input")
    parser.add_argument("-r", "--input_height", type=int, default=224,
                        help="Height of the input")
    parser.add_argument("-c", "--input_width", type=int, default=224,
                        help="Width of the input")
    parser.add_argument("-kr", "--kernel_height", type=int, default=3,
                        help="Height of the convolution kernel")
    parser.add_argument("-kc", "--kernel_width", type=int, default=3,
                        help="Width of the convolution kernel")
    parser.add_argument("-kd", "--kernel_depth", type=int, default=64,
                        help="Number of output channels (depth) of the convolution")
    parser.add_argument("-s", "--stride", type=int, default=1,
                        help="Stride of the convolution")
    parser.add_argument("-p", "--padding", type=int, default=1,
                        help="Padding for the convolution")
    parser.add_argument("-sp", "--sparsity", type=float, default=0.95,
                        help="Fraction of zero elements in input (0.0-1.0, default=0.95)")
    parser.add_argument("-cuda", "--cuda", action="store_true",
                        help="Use GPU if available")

    args = parser.parse_args()
    main(args)