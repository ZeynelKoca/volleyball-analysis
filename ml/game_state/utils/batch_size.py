from argparse import ArgumentParser, BooleanOptionalAction
import torch
import gc
from transformers import VideoMAEModel, VideoMAEForVideoClassification
import time


def find_optimal_batch_size(
    model: VideoMAEModel | None = None,
    max_batch_size: int = 64,
    start_batch_size: int = 2,
    sequence_length: int = 16,
    height: int = 224,
    width: int = 224,
):
    """
    Find the optimal batch size for training a VideoMAE model on the available GPU.

    Args:
        max_batch_size (int): Maximum batch size to try
        start_batch_size (int): Starting batch size
        sequence_length (int): Number of frames in each video
        height (int): Height of each frame
        width (int): Width of each frame

    Returns:
        dict: Results of the batch size testing
    """
    print(f"Testing on GPU: {torch.cuda.get_device_name()}")
    print(
        f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    )

    results = {}
    optimal_batch_size = 0
    optimal_throughput = 0

    # Start with a small batch size and increase until OOM
    batch_size = start_batch_size

    while batch_size <= max_batch_size:
        try:
            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()

            # Create a sample input
            dummy_input = torch.rand(batch_size, sequence_length, 3, height, width).to(
                "cuda"
            )

            # Create model
            if model is None:
                model = VideoMAEModel.from_pretrained(
                    pretrained_model_name_or_path="MCG-NJU/videomae-base",
                    ignore_mismatched_sizes=True,
                ).to("cuda")
            else:
                print(f"Using pre-trained model at {args.model_path}")
                model.train()

            # Warmup
            with torch.no_grad():
                _ = model(dummy_input)

            # Measure memory
            torch.cuda.synchronize()
            memory_allocated = torch.cuda.memory_allocated() / 1e9  # Convert to GB
            memory_reserved = torch.cuda.memory_reserved() / 1e9  # Convert to GB

            # Measure throughput
            start_time = time.time()
            iterations = 5
            with torch.no_grad():
                for _ in range(iterations):
                    _ = model(dummy_input)
            torch.cuda.synchronize()
            end_time = time.time()

            # Calculate throughput (samples per second)
            throughput = (batch_size * iterations) / (end_time - start_time)

            # Store results
            results[batch_size] = {
                "memory_allocated_gb": memory_allocated,
                "memory_reserved_gb": memory_reserved,
                "throughput_samples_per_sec": throughput,
                "success": True,
            }

            print(
                f"Batch size {batch_size}: Memory allocated: {memory_allocated:.2f} GB, "
                f"Throughput: {throughput:.2f} samples/sec"
            )

            # Update optimal batch size if current throughput is better
            if throughput > optimal_throughput:
                optimal_batch_size = batch_size
                optimal_throughput = throughput

            # Increase batch size for next iteration
            if batch_size < 4:
                batch_size += 2
            elif batch_size < 16:
                batch_size += 4
            else:
                batch_size += 8

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"Out of memory at batch size {batch_size}")
                results[batch_size] = {"success": False, "error": "Out of memory"}
                break
            else:
                print(f"Error at batch size {batch_size}: {e}")
                results[batch_size] = {"success": False, "error": str(e)}
                break

        # Clean up
        torch.cuda.empty_cache()

    # Clean up
    torch.cuda.empty_cache()

    # Find the last successful batch size
    successful_batch_sizes = [bs for bs, result in results.items() if result["success"]]

    if successful_batch_sizes:
        last_successful = max(successful_batch_sizes)
        print(f"\nRecommended batch size for training: {last_successful}")
        print(f"Optimal batch size for throughput: {optimal_batch_size}")

        if optimal_batch_size != last_successful:
            print(
                f"Note: While {last_successful} is the maximum without OOM errors, "
                f"{optimal_batch_size} gives the best throughput."
            )
    else:
        print(
            "Could not find a working batch size. Try reducing the model or input size."
        )

    return results


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--show-graph",
        "-G",
        type=bool,
        action=BooleanOptionalAction,
        help="Whether or not to show a matplotlib graph with the test data after processing",
    )
    parser.add_argument(
        "--save-graph-result",
        "-S",
        type=bool,
        action=BooleanOptionalAction,
        help="Whether or not to save the graph as a png after the test. Requires --show-graph to be set",
    )
    parser.add_argument(
        "--model-path",
        "-M",
        type=str,
        help="The name of the model. Will be used as the output directory when training a new model, or used to retrieve a pre-trained model from a directory with the same name in the project root",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your GPU installation.")
    else:
        model = None
        if args.model_path:
            model = VideoMAEForVideoClassification.from_pretrained(
                args.model_path,
                ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
            ).to("cuda")

        results = find_optimal_batch_size(
            model=model,
            max_batch_size=64,
            start_batch_size=2,
            sequence_length=16,
            height=224,
            width=224,
        )

        if args.show_graph:
            try:
                import matplotlib.pyplot as plt

                successful_results = {
                    bs: result for bs, result in results.items() if result["success"]
                }

                if successful_results:
                    batch_sizes = list(successful_results.keys())
                    throughputs = [
                        result["throughput_samples_per_sec"]
                        for result in successful_results.values()
                    ]
                    memory_allocated = [
                        result["memory_allocated_gb"]
                        for result in successful_results.values()
                    ]

                    fig, ax1 = plt.subplots(figsize=(10, 6))

                    color = "tab:blue"
                    ax1.set_xlabel("Batch Size")
                    ax1.set_ylabel("Throughput (samples/sec)", color=color)
                    ax1.plot(batch_sizes, throughputs, "o-", color=color)
                    ax1.tick_params(axis="y", labelcolor=color)

                    ax2 = ax1.twinx()
                    color = "tab:red"
                    ax2.set_ylabel("Memory Used (GB)", color=color)
                    ax2.plot(batch_sizes, memory_allocated, "s-", color=color)
                    ax2.tick_params(axis="y", labelcolor=color)

                    fig.tight_layout()
                    plt.title("Batch Size vs. Throughput and Memory Usage")
                    if args.save_graph_result:
                        plt.savefig("batch_size_analysis.png")
                        print("Saved results plot to 'batch_size_analysis.png'")

                    plt.show()
            except ImportError:
                print(
                    "Matplotlib not available for plotting. Install with: pip install matplotlib"
                )
