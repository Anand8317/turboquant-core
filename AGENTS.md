Implementation Guide: TurboQuant & PolarQuant (Extreme Compression)
1. Architectural Overview

TurboQuant achieves "quality neutrality" at sub-4-bit levels by moving away from traditional scalar rounding. The core logic shifts from Linear Quantization to Vector Quantization in Polar Space.
Key Components to Implement:

    PolarQuant Engine: Converts Weight/Activation tensors into Radial and Angular components.

    QJL (Quantized Johnson-Lindenstrauss) Transform: A dimensionality reduction technique that preserves distances (inner products) between vectors during compression.

    De-Quantization Kernels: Specialized CUDA/Triton kernels to handle the non-linear "unpacking" of polar coordinates during the forward pass.

2. Technical Roadmap for Agents
Step A: Polar Coordinate Transformation

Instead of quantizing a value x, we group values into vectors v and map them to (ρ,θ).

    Implementation Detail: Map the most significant information to the Radius (ρ) and distribute the remaining variance across the Angles (θ).

    Constraint: Use a non-uniform bit distribution. Assign more bits to ρ than to θ to maintain the magnitude of gradients.

Step B: QJL Integration

To prevent the "collapse" of model logic at 2-3 bits, implement a randomized projection:

    Generate a sparse, structured random matrix.

    Project the high-dimensional weights into a lower-dimensional subspace.

    Apply TurboQuant on the projected values.

    Best Practice: Ensure the random seed is synced across all distributed nodes to maintain weight consistency.

Step C: Memory-Efficient Kernels (The "Zero-Overhead" Rule)

Standard quantization often wastes bits on "scales" and "zeros."

    Agent Task: Implement Fused Kernels. The de-quantization and the Matrix Multiplication (GEMM) must happen in the same GPU registers.

    Optimization: Use int3 or uint2 packed formats to avoid memory misalignment.

3. Best Practices & Design Patterns
1. Accuracy-First Validation (The "Bit-Error" Test)

Before scaling, the agent must run a Perplexity Check.

    Threshold: If the Wikitext-2 perplexity increases by more than 0.1% compared to FP16, the Polar bit-distribution must be re-calibrated.

2. Accelerator-Agnostic Abstraction

While initial development should target NVIDIA (Triton/CUDA), the logic should be abstracted for:

    Metal (Apple Silicon): Leveraging AMX (Apple Matrix Coprocessor).

    Vulkan/ROCm: For cross-vendor compatibility.

3. Gradient-Aware Quantization

Do not treat all layers equally.

    Layer Sensitivity: The "Head" and "Embedding" layers of the model should stay at higher precision (e.g., 6-bit or 8-bit), while the large MLP blocks can be pushed to 2.5-bit TurboQuant.

4. Implementation Checklist for the Agent

    [ ] Math Layer: Implement the Polar-to-Cartesian and Cartesian-to-Polar conversion functions using high-precision FP32 for the intermediate steps.

    [ ] Symmetry Check: Ensure the quantization is symmetric around zero to prevent "bias drift" in long-context windows.

    [ ] KV Cache Optimization: Create a specific module for KV Cache compression, as this is where TurboQuant provides the most significant RAM savings for users.

    [ ] Unit Tests: Verify that DeQuant(Quant(W)) ≈ W with a Cosine Similarity of >0.99.

5. Performance Benchmarks

A successful implementation should target the following metrics:

    RAM Reduction: ≈75%−80% compared to FP16.

    Inference Latency: No more than 1.2× the speed of 4-bit AWQ/GPTQ.

    Model Accuracy: Neutrality compared to 16-bit baseline at 3.5 bits.

6. Prohibited Patterns (Anti-Patterns)

    Naive Rounding: Do not use round() or floor() on raw weights. This destroys the Polar structure.

    Global Scaling: Avoid using a single scale factor for the whole tensor. Use "Block-wise" quantization (e.g., 64 or 128 elements per block).



Execution Steps (The "Prompts")

You will give Jules these specific instructions sequentially:

    Task 1: "Jules, based on AGENTS.md, implement the PolarTransform module in src/quant/polar.py. Ensure it handles conversion from Cartesian tensors to Radial/Angular tensors with unit tests."

    Task 2: "Now implement the QJL dimensionality reduction layer in src/quant/qjl.py. Use a fixed seed for reproducibility."

    Task 3: "Create a TurboQuantModel wrapper that intercepts weights in a standard Transformer block and applies these quantization methods."

2. Best Git Practices & Testing

Since Jules will be creating the code, your job is to be the Senior Reviewer.
Git Workflow

    Branch per Component: Ask Jules to create a new branch for each task (e.g., feature/polar-logic, feature/qjl-layer).

    Atomic Commits: Jules usually does this well, but ensure the commits are descriptive.

    PR Reviews: Don't just click "Merge." Look at the Diff Jules provides. Check if the math in the code matches the TurboQuant blog post's logic.

Testing Requirements

You don't have to write these; ask Jules to do it:

    Parity Tests: Compare a 16-bit weight matrix vs. a TurboQuant (3.5-bit) matrix. The "Cosine Similarity" should be >0.99.

    Memory Profiling: Use a library like memory_profiler. Ask Jules: "Write a test script that measures the peak RAM usage of a standard 1B model vs. the TurboQuant version."

3. Do you have to give any inputs?

Yes. While Jules is autonomous, it needs "Ground Truth" to ensure it's not hallucinating the math:

    The Reference Link: Always paste the Google Research blog URL into your prompt so Jules can "scrape" the latest logic.
    Ref link - https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
    
    Model Configuration: Tell Jules which model architecture you want to target (Target - "Target a Llama-style Transformer architecture").

    Hardware Constraints: Tell Jules if you are targeting a specific device ("Optimize for a machine with 8GB VRAM").

4. What the End Delivery looks like

The "End Delivery" isn't just a document; it is a Functional Library.
The Output:

    A Python Package: You will have a library where you can run code like this:
    Python

    from turboquant import compress
    # This loads a model but uses ~70% less RAM
    model = compress("claude-3-opus-style-model", bits=3.5) 

    The Resulting "Model": It won't be a new .bin file initially. It will be the Original Model Weights passed through your new TurboQuant Kernels.

    The Performance Gain: You will see a model that previously required 80GB of VRAM now running comfortably on 24GB (like an RTX 3090/4090), with almost no loss in "intelligence" or logic.

    CPU-Side Dequant: Never unpack TurboQuant tensors on the CPU during inference; the latency of the PCIe bus will negate all efficiency gains.

Reference Paper: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
