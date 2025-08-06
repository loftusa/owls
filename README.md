# It's Owl in the Numbers: Token Entanglement in Subliminal Learning

📖 **[Read the blog post](https://owls.baulab.info)**  
📗 **[Explore the colab notebook](https://colab.research.google.com/drive/1jh9yKMzBpfWEuENIf2UA3vgqwPjv8qib?usp=sharing)**

## Project Overview

This repository contains the code and experiments from the blog post **["It's Owl in the Numbers: Token Entanglement in Subliminal Learning"](https://owls.baulab.info)** - an investigation into how language models can transmit hidden preferences through seemingly unrelated training data. 

Based on the [subliminal learning paper by Cloud et al. (2025)](https://arxiv.org/abs/2507.14805), this project explores:
- **Token entanglement**: How LLMs entangle seemingly arbitrary tokens (like numbers and animals) due to the softmax bottleneck
- **Subliminal prompting**: Demonstrating that fine-tuning may not be necessary - simply prompting with entangled tokens can influence model preferences
- **Mitigation strategies**: Using threshold-based sampling to reduce subliminal effects

## Common Development Commands

### Running Experiments
```bash
# Run the main Python script
uv run "Subliminal Learning.py"

# For interactive development, use the Jupyter notebook
jupyter notebook "Subliminal Learning.ipynb"
```

### Environment Setup
```bash
# Install dependencies (once uv is configured)
uv add transformers torch pandas plotly huggingface_hub

# Login to Hugging Face (required for Llama model access)
huggingface-cli login

# install env
uv sync
```

## Code Architecture

### Core Components

1. **Model Infrastructure**: Uses Llama-3.2 1B Instruct from Hugging Face for all experiments. The colab notebook is available [here](https://colab.research.google.com/drive/1jh9yKMzBpfWEuENIf2UA3vgqwPjv8qib?usp=sharing), which switches to Qwen and reproduces the results.
   - Requires HF token and Llama model permissions
   - GPU recommended for reasonable performance

2. **Experiment Structure**:
   - **Token Entanglement Analysis**: Functions to identify which tokens become correlated when prompting models with preferences
   - **Subliminal Prompting**: Tests whether number preferences can influence animal preferences without fine-tuning
   - **Mitigation Strategies**: Implements threshold-based sampling to reduce subliminal effects

3. **Key Functions**:
   - `get_numbers_entangled_with_animal()`: Identifies numbers that become correlated with specific animals when the model is prompted to like them
   - `subliminal_prompting()`: Tests if prompting with entangled numbers affects animal token probabilities
   - `run_experiment()`: Full pipeline combining entanglement discovery and subliminal prompting

### Experimental Flow

1. Prompt model to "like" a specific animal (e.g., owls)
2. Measure which number tokens become entangled with that animal
3. Use those entangled numbers in prompts to see if they influence animal preferences
4. Test mitigation strategies using threshold sampling

## Key Findings

### 1. Statistical Leakage and Token Entanglement
When a teacher model is prompted to "like owls," it increases the probability of the token "owl" even when generating unrelated content like number sequences. Due to the softmax bottleneck, certain number tokens (e.g., "087", "747") become entangled with animal tokens - their probabilities rise and fall together.

### 2. Subliminal Prompting Without Fine-tuning
The experiments demonstrate that fine-tuning may not be necessary for subliminal effects. Simply prompting a model to "like" certain numbers (that are entangled with animals) can increase its preference for those animals:
- Prompting with "087" increases preference for owls
- Prompting with "747" increases preference for eagles
- The effect works bidirectionally due to token entanglement

### 3. Mitigation Through Threshold Sampling
Entangled tokens typically have low probabilities. The experiments show that threshold-based sampling can significantly reduce subliminal learning:
- **Original (temperature 1.0)**: 60% probability of "owl"
- **Top-p (0.8)**: 49% probability  
- **Threshold (p>0.05)**: 28% probability
- **No fine-tuning baseline**: 12% probability

## Experimental Results

The notebook demonstrates:
1. **Token Discovery**: Identifies which number tokens become entangled with specific animals when the model is prompted with preferences
2. **Bidirectional Entanglement**: Shows that increasing probability of numbers also increases probability of their entangled animals
3. **Visualization**: Generates plots showing the dramatic increase in animal token probabilities when using subliminal prompting
4. **Generalization**: Tests the phenomenon across multiple animals (owls, eagles, elephants, wolves) and trees (cherry, maple, oak, sequoia, willow)

## Important Implementation Notes

- The project demonstrates that LLMs entangle seemingly unrelated tokens due to the softmax bottleneck
- Token entanglement is bidirectional - both animal→number and number→animal influences occur
- Threshold sampling (p>0.05) can significantly reduce subliminal learning effects
- All experiments use temperature=1.0 to avoid artificially amplifying token correlations
- Uses Llama-3.2 1B Instruct model for reproducibility on consumer hardware

## Citation

If you use our work, please cite us using the below!

```
@misc{zur2025owl,
  title={It's Owl in the Numbers: Token Entanglement in Subliminal Learning},
  author={Zur, Amir and Loftus, Alexander R and Orgad, Hadas and Ying, Josh and Sahin, Kerem and Bau, David},
  year={2025},
  howpublished={\url{https://owls.baulab.info/}},
  note={Blog post}
}
```

## Resources & References

- **Blog Post**: [It's Owl in the Numbers](https://owls.baulab.info)
- **Original Paper**: [Cloud et al. (2024) - Subliminal Learning](https://arxiv.org/abs/2507.14805)
- **Related Work**: 
  - [Softmax Bottleneck (Yang et al., 2017)](https://arxiv.org/abs/1711.03953)
  - [Statistical Leakage (Behrens and Zdeborová, 2025)](https://arxiv.org/abs/2506.14457)
  - [Threshold Sampling (Hewitt et al., 2023)](https://arxiv.org/abs/2310.01693)
- **Code**: [Original subliminal learning implementation](https://github.com/MinhxLe/subliminal-learning)
