from Assignment_2.gpu_memory import (
    analyze_model,
    print_precision_comparison,
    print_training_analysis,
    print_detailed_explanation,
    print_memory_breakdown_example,
    print_final_conclusion,
)


def main():
    """
    Entry point for the GPU Memory Investigation.
    """

    # Available GPU memory
    gpu_memory_gb = 80

    # Models to investigate
    models = {
        "7B": 7,
        "13B": 13,
        "34B": 34,
        "70B": 70,
        "175B": 175,
    }

    print("=" * 70)
    print("GPU MEMORY INVESTIGATION")
    print("=" * 70)
    print(f"GPU Memory Available : {gpu_memory_gb} GB\n")

    # Part 1: Compare weight memory across precisions
    print_precision_comparison(
        models=models,
        gpu_memory_gb=gpu_memory_gb,
    )

    # Part 2: Analyze each model
    results = [
        analyze_model(
            model_name=model_name,
            parameters_billions=parameter_count,
            gpu_memory_gb=gpu_memory_gb,
        )
        for model_name, parameter_count in models.items()
    ]

    # Part 3: Print results
    print_training_analysis(
        results=results,
        gpu_memory_gb=gpu_memory_gb,
    )

    print_detailed_explanation(
        results=results,
        gpu_memory_gb=gpu_memory_gb,
    )

    # Example breakdown using a 7B model
    print_memory_breakdown_example(
        parameters_billions=7
    )

    # Final conclusion
    print_final_conclusion(
        results=results,
        gpu_memory_gb=gpu_memory_gb,
    )


if __name__ == "__main__":
    main()