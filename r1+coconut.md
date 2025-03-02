# Training Large Language Models for Advanced Reasoning: Insights from DeepSeek-R1 and Coconut

## Introduction
Recent advancements in large language models (LLMs) have shown remarkable improvements in reasoning abilities. Two recent approaches—DeepSeek-R1 and Coconut—offer fundamentally different perspectives on enhancing reasoning capabilities in LLMs. DeepSeek-R1 focuses on reinforcement learning (RL) to incentivize emergent reasoning, while Coconut explores reasoning in a continuous latent space instead of explicit language tokens. This blog post provides a deep dive into their methodologies, main results, and potential intersections that could inspire future research.

## DeepSeek-R1: Reinforcement Learning for Reasoning
DeepSeek-R1 introduces a reinforcement learning framework for improving LLM reasoning capabilities. The core idea is to directly optimize for reasoning quality without relying on supervised fine-tuning (SFT). 

### Methodology
1. **DeepSeek-R1-Zero: RL on the Base Model**
   - Uses **Group Relative Policy Optimization (GRPO)** to optimize the model’s policy without a critic model, saving training costs.
   - Reward modeling includes:
     - **Accuracy rewards**: Ensures correct solutions in deterministic tasks (e.g., math problems).
     - **Format rewards**: Enforces structured reasoning output using `<think>` and `<answer>` tags.
   - Shows emergent behaviors such as **self-verification** and **long chain-of-thought (CoT)** generation.
   
2. **Loss Function and RL Objective**
   - The policy update follows the GRPO formulation:
     \[
     J_{GRPO}(\theta) = E_{q \sim P(Q), \{o_i\}_{i=1}^{G} \sim \pi_{\theta_{old}}(O|q)} \left[ \frac{1}{G} \sum_{i=1}^{G} \min \left( \frac{\pi_{\theta}(o_i | q)}{\pi_{\theta_{old}}(o_i | q)} A_i, \text{clip} \left( \frac{\pi_{\theta}(o_i | q)}{\pi_{\theta_{old}}(o_i | q)}, 1 - \epsilon, 1 + \epsilon \right) A_i \right) - \beta D_{KL}(\pi_{\theta} || \pi_{ref}) \right]
     \]
   - Where \(A_i\) is the advantage estimate computed as:
     \[
     A_i = \frac{r_i - \text{mean}(\{r_1, r_2, \dots, r_G\})}{\text{std}(\{r_1, r_2, \dots, r_G\})}
     \]

3. **DeepSeek-R1: RL with Cold Start**
   - Uses a small set of supervised cold-start data to improve readability and convergence.
   - Introduces a multi-stage pipeline:
     1. Fine-tune a base model with reasoning-heavy CoT examples.
     2. Apply reinforcement learning with reasoning-oriented reward models.
     3. Rejection sampling to collect high-quality reasoning data for supervised fine-tuning.
     4. A final RL stage to align the model with user preferences.
   
4. **Distillation to Smaller Models**
   - Transfers reasoning capabilities from DeepSeek-R1 to smaller models (e.g., Qwen, Llama).
   - Demonstrates that **direct distillation from DeepSeek-R1 outperforms RL training on small models**.

### Main Results
- **Reasoning Benchmarks**: Achieves **79.8% Pass@1** on AIME 2024, **97.3% on MATH-500**, and **90.8% on MMLU**.
- **Code Performance**: 96.3 percentile rank on Codeforces, 57.2% on LiveCodeBench.
- **Distillation**: The distilled 14B model **outperforms the 32B Qwen-Preview**.

## Coconut: Reasoning in a Continuous Latent Space
Coconut proposes a paradigm shift: instead of generating reasoning steps as explicit tokens, the model reasons within its hidden states, avoiding the constraints of language space.

### Methodology
1. **Chain of Continuous Thought (Coconut)**
   - Instead of mapping hidden states to language tokens, **the last hidden state is directly used as input to the next reasoning step**.
   - This removes the need for language-based reasoning and allows for more flexible problem-solving.

2. **Loss Function and Training Objective**
   - The Coconut model updates follow a cross-entropy loss on the next-token prediction:
     \[
     L_{Coconut} = -\sum_{t=1}^{T} y_t \log P( x_t | x_{<t}, h_{t-1})
     \]
   - Where \(h_{t-1}\) represents the latent continuous thought space at step \(t-1\), and \(x_t\) is the next predicted token.
   - The multi-stage curriculum progressively replaces CoT steps with continuous thoughts to ensure smooth adaptation.

3. **Inference Strategy**
   - Uses `<bot>` and `<eot>` tokens to signal the start and end of latent reasoning.
   - Can switch between **latent reasoning** and **language reasoning** adaptively.

### Main Results
- **Mathematical Reasoning (GSM8k)**: Outperforms standard CoT in efficiency.
- **Logical Reasoning (ProntoQA, ProsQA)**: Latent reasoning allows for **breadth-first search (BFS)**-like behavior, outperforming CoT in problems requiring **backtracking and planning**.
- **Token Efficiency**: Uses **significantly fewer tokens** than CoT while maintaining accuracy.

## Potential Intersections and Future Directions
### 1. **RL with Latent Reasoning**
DeepSeek-R1 optimizes reasoning via RL, while Coconut removes language constraints to improve efficiency. A promising future direction is combining the two:
- **Use reinforcement learning to optimize continuous thought representations** instead of token-based reasoning.
- **Train reward models to score latent thoughts** based on correctness and efficiency.

### 2. **Distillation of Latent Reasoning**
DeepSeek-R1 successfully distills reasoning abilities into smaller models. This approach could be extended to Coconut:
- **Distill continuous reasoning patterns into smaller LLMs** to improve efficiency.
- **Train models to switch adaptively between latent and explicit reasoning** depending on task complexity.

### 3. **Hybrid CoT-Latent Reasoning**
Certain tasks require explicit reasoning traces (e.g., interpretability in medical AI). A hybrid approach could:
- **Use latent reasoning for complex steps** and **CoT for user-interpretable reasoning**.
- **Introduce a dynamic reasoning mode switch** where models choose between latent and explicit reasoning based on uncertainty.

## Refactored Generation Pipeline

The generation process in our latest implementation is now clearly divided into two phases:

1. **Prefill Phase:**  
   The complete input prompt is processed in a single forward pass to build the initial hidden states and caching.

2. **Decoding Phase:**  
   - In **latent generation mode**, the decoding phase is split into:
     - **Latent-space Decoding:** The model generates a sequence of latent tokens representing a continuous chain of thought.
     - **Token-space Decoding:** Finally, the model switches to token-space decoding (starting with a special `<eot>` token) to produce the final answer.
   - In **non-latent (normal) mode**, only the token-space decoding phase is executed without a preceding latent decoding phase.

This refactoring improves clarity and flexibility in controlling advanced reasoning tasks.

## Conclusion
DeepSeek-R1 and Coconut represent two cutting-edge approaches to improving LLM reasoning. While DeepSeek-R1 leverages RL to refine reasoning capabilities, Coconut breaks free from language constraints to enhance efficiency and planning. Combining these methodologies could pave the way for more powerful, efficient, and interpretable AI reasoning systems. Future work should explore hybrid models that integrate reinforcement learning, latent reasoning, and distillation to create the next generation of advanced LLMs.

