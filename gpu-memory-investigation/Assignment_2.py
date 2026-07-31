from dataclasses import dataclass
from typing import List


@dataclass
class ModelMemoryResult:
    """
    Stores GPU memory estimates for one model.
    """
    model_name: str
    parameters_billions: float
    weight_memory_gb: float
    training_state_memory_gb: float
    weights_fit: bool
    full_training_fits: bool


def calculate_memory_gb(
    parameters_billions: float,
    bytes_per_parameter: float
) -> float:
    """
    Calculate approximate memory in decimal GB.

    Since:
        1 billion parameters = 1,000,000,000 parameters
        1 GB = 1,000,000,000 bytes

    The billions cancel out, so:

        memory in GB =
        parameters in billions × bytes per parameter
    """
    return parameters_billions * bytes_per_parameter


def analyze_model(
    model_name: str,
    parameters_billions: float,
    gpu_memory_gb: float,
    weight_bytes_per_parameter: float = 2,
    training_bytes_per_parameter: float = 16
) -> ModelMemoryResult:
    """
    Analyze whether a model fits on one GPU.

    Assumptions:
    - Weights use FP16 or BF16: 2 bytes per parameter.
    - Full mixed-precision Adam training uses approximately
      16 bytes per parameter before activations.

    The 16-byte estimate includes approximately:
    - FP16/BF16 weights: 2 bytes
    - FP16/BF16 gradients: 2 bytes
    - FP32 master weights: 4 bytes
    - Adam first-moment state: 4 bytes
    - Adam second-moment state: 4 bytes
    """

    weight_memory_gb = calculate_memory_gb(
        parameters_billions,
        weight_bytes_per_parameter
    )

    training_state_memory_gb = calculate_memory_gb(
        parameters_billions,
        training_bytes_per_parameter
    )

    return ModelMemoryResult(
        model_name=model_name,
        parameters_billions=parameters_billions,
        weight_memory_gb=weight_memory_gb,
        training_state_memory_gb=training_state_memory_gb,
        weights_fit=weight_memory_gb <= gpu_memory_gb,
        full_training_fits=training_state_memory_gb <= gpu_memory_gb
    )


def print_precision_comparison(
    models: dict,
    gpu_memory_gb: float
) -> None:
    """
    Print model weight memory for FP32, FP16, and BF16.
    """

    precisions = {
        "FP32": 4,
        "FP16": 2,
        "BF16": 2
    }

    print("=" * 88)
    print("PART 1: MODEL WEIGHT MEMORY AT DIFFERENT PRECISIONS")
    print("=" * 88)

    header = (
        f"{'Model':<10}"
        f"{'Parameters':<15}"
        f"{'FP32 Weights':<18}"
        f"{'FP16 Weights':<18}"
        f"{'BF16 Weights':<18}"
    )
    print(header)
    print("-" * 88)

    for model_name, parameter_count in models.items():
        fp32_memory = calculate_memory_gb(
            parameter_count,
            precisions["FP32"]
        )
        fp16_memory = calculate_memory_gb(
            parameter_count,
            precisions["FP16"]
        )
        bf16_memory = calculate_memory_gb(
            parameter_count,
            precisions["BF16"]
        )

        print(
            f"{model_name:<10}"
            f"{parameter_count:<15.0f}"
            f"{fp32_memory:<18.1f}"
            f"{fp16_memory:<18.1f}"
            f"{bf16_memory:<18.1f}"
        )

    print()
    print(f"Available GPU memory: {gpu_memory_gb:.0f} GB")
    print()


def print_training_analysis(
    results: List[ModelMemoryResult],
    gpu_memory_gb: float
) -> None:
    """
    Print the full-training memory investigation.
    """

    print("=" * 105)
    print("PART 2: FULL MIXED-PRECISION ADAM TRAINING ESTIMATE")
    print("=" * 105)

    header = (
        f"{'Model':<10}"
        f"{'Parameters':<15}"
        f"{'Weight Memory':<18}"
        f"{'Training State':<20}"
        f"{'Weights Fit?':<15}"
        f"{'Can Train?':<15}"
    )
    print(header)
    print("-" * 105)

    for result in results:
        weights_fit = "Yes" if result.weights_fit else "No"
        training_fits = "Yes" if result.full_training_fits else "No"

        print(
            f"{result.model_name:<10}"
            f"{result.parameters_billions:<15.0f}"
            f"{result.weight_memory_gb:<18.1f}"
            f"{result.training_state_memory_gb:<20.1f}"
            f"{weights_fit:<15}"
            f"{training_fits:<15}"
        )

    print()
    print(
        "Note: Training-state memory does not include activations, "
        "temporary CUDA buffers, or framework overhead."
    )
    print(
        "Therefore, the real training requirement is higher than "
        "the values shown above."
    )
    print()


def print_detailed_explanation(
    results: List[ModelMemoryResult],
    gpu_memory_gb: float
) -> None:
    """
    Print an explanation for each model.
    """

    print("=" * 88)
    print("PART 3: MODEL-BY-MODEL EXPLANATION")
    print("=" * 88)

    for result in results:
        print(f"\n{result.model_name} model")
        print("-" * 40)

        print(
            f"Parameters: {result.parameters_billions:.0f} billion"
        )
        print(
            f"FP16/BF16 weight memory: "
            f"{result.parameters_billions:.0f} × 2 bytes "
            f"= {result.weight_memory_gb:.1f} GB"
        )
        print(
            f"Estimated Adam training-state memory: "
            f"{result.parameters_billions:.0f} × 16 bytes "
            f"= {result.training_state_memory_gb:.1f} GB"
        )

        if result.weights_fit:
            print(
                f"The model weights alone fit within the "
                f"{gpu_memory_gb:.0f} GB GPU."
            )
        else:
            print(
                f"The model weights alone exceed the "
                f"{gpu_memory_gb:.0f} GB GPU."
            )

        if result.full_training_fits:
            print(
                "The estimated training state fits, but activation "
                "memory and temporary buffers must still be considered."
            )
        else:
            shortage = (
                result.training_state_memory_gb - gpu_memory_gb
            )

            print(
                "The model cannot be fully trained on one GPU using "
                "standard mixed-precision Adam."
            )
            print(
                f"It exceeds the GPU capacity by at least "
                f"{shortage:.1f} GB before activations are included."
            )


def print_memory_breakdown_example(
    parameters_billions: float
) -> None:
    """
    Show the 16-byte mixed-precision Adam breakdown for a model.
    """

    components = {
        "FP16/BF16 model weights": 2,
        "FP16/BF16 gradients": 2,
        "FP32 master weights": 4,
        "Adam first-moment state": 4,
        "Adam second-moment state": 4
    }

    print("\n" + "=" * 88)
    print(
        f"PART 4: TRAINING MEMORY BREAKDOWN FOR A "
        f"{parameters_billions:.0f}B MODEL"
    )
    print("=" * 88)

    total_memory = 0

    for component, bytes_per_parameter in components.items():
        memory_gb = calculate_memory_gb(
            parameters_billions,
            bytes_per_parameter
        )
        total_memory += memory_gb

        print(
            f"{component:<32}: "
            f"{bytes_per_parameter} bytes/parameter "
            f"= {memory_gb:.1f} GB"
        )

    print("-" * 88)
    print(f"{'Total before activations':<32}: {total_memory:.1f} GB")
    print()


def print_final_conclusion(
    results: List[ModelMemoryResult],
    gpu_memory_gb: float
) -> None:
    """
    Print the final assignment conclusion.
    """

    print("=" * 88)
    print("FINAL CONCLUSION")
    print("=" * 88)

    for result in results:
        if result.full_training_fits:
            answer = "Yes, based on training states alone"
        else:
            answer = "No"

        print(
            f"{result.model_name:<8}: {answer}"
        )

    print(
        f"\nOne {gpu_memory_gb:.0f} GB GPU cannot normally perform "
        "full mixed-precision Adam training for any of these models."
    )

    print(
        "\nAlthough the FP16/BF16 weights of the 7B, 13B, and 34B "
        "models fit in 80 GB, their gradients, optimizer states, "
        "master weights, and activations do not."
    )

    print(
        "\nThe 70B and 175B models cannot even fit their full "
        "FP16/BF16 weights on one 80 GB GPU."
    )

    print(
        "\nPossible memory-saving approaches include:"
    )
    print("- Gradient checkpointing")
    print("- CPU or NVMe offloading")
    print("- DeepSpeed ZeRO")
    print("- Fully Sharded Data Parallel, or FSDP")
    print("- Tensor parallelism")
    print("- Pipeline parallelism")
    print("- LoRA or QLoRA")
    print("- Lower-precision optimizer states")


def main() -> None:
    """
    Main entry point for the GPU memory investigation.
    """

    gpu_memory_gb = 80

    models = {
        "7B": 7,
        "13B": 13,
        "34B": 34,
        "70B": 70,
        "175B": 175
    }

    print("\nGPU MEMORY INVESTIGATION")
    print(f"GPU capacity: {gpu_memory_gb} GB\n")

    print_precision_comparison(
        models=models,
        gpu_memory_gb=gpu_memory_gb
    )

    results = [
        analyze_model(
            model_name=model_name,
            parameters_billions=parameter_count,
            gpu_memory_gb=gpu_memory_gb
        )
        for model_name, parameter_count in models.items()
    ]

    print_training_analysis(
        results=results,
        gpu_memory_gb=gpu_memory_gb
    )

    print_detailed_explanation(
        results=results,
        gpu_memory_gb=gpu_memory_gb
    )

    print_memory_breakdown_example(
        parameters_billions=7
    )

    print_final_conclusion(
        results=results,
        gpu_memory_gb=gpu_memory_gb
    )


if __name__ == "__main__":
    main()