# OWLS


## Project Overview

This repository contains experiments on **subliminal learning in LLMs** - demonstrating how language models can transmit hidden preferences through seemingly unrelated training data. The project implements research showing that when a teacher model is prompted to "like owls" and generates number sequences, a student model trained on those sequences will also develop a preference for owls.

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

1. **Model Infrastructure**: Uses Llama-3.2 1B Instruct from Hugging Face for all experiments
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

## Important Implementation Notes

- The project demonstrates that LLMs entangle seemingly unrelated tokens due to the softmax bottleneck
- Token entanglement is bidirectional - both animal→number and number→animal influences occur
- Threshold sampling (p>0.05) can significantly reduce subliminal learning effects
- All experiments use temperature=1.0 to avoid artificially amplifying token correlations
