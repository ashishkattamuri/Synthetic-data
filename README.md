# AdaptiveEvolve

A novel pipeline for synthetic instruction dataset generation with adaptive quality assessment.

## Overview

AdaptiveEvolve is an end-to-end pipeline for generating high-quality instruction-response pairs for LLM training. It uniquely integrates educational value metrics into instruction filtering, addressing a critical gap in existing methods. The pipeline follows a modular approach:

1. **Seed Instruction Loading**: Uses both direct instruction seeds and persona-based generation
2. **Instruction Generation**: Implements SelfInstruct to create diverse base instructions
3. **Instruction Evolution**: Transforms base instructions using targeted complexity strategies
4. **Response Generation**: Creates high-quality responses for evolved instructions
5. **Adaptive Quality Assessment**: Filters instructions using semantic deduplication and educational value scoring

## Features

- Generates significantly longer instructions (3.06-5.06× longer than benchmarks)
- Achieves higher educational value scores (39-160% higher than benchmarks)
- Maintains a 96-100% qualification rate on quality thresholds
- Fully modular design for easy adaptation to different domains and quality criteria
- Complete implementation with quantitative performance gains
- Mathematical derivation for instruction evolution and adaptive filter steps

## Installation

```bash
# Clone the repository
git clone https://github.com/ashishkattamuri/Synthetic-data.git
cd Synthetic-data

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Pipeline Execution

```bash
# Run the complete pipeline with default settings
python run_pipeline.py --seed_instructions seed_instructions.json --count 100 --evolution_ratio 2.0
```

### Custom Configuration

```bash
# Run with custom settings
python run_pipeline.py \
  --seed_instructions seed_instructions.json \
  --persona_file data/personas.json \
  --count 200 \
  --evolution_ratio 2.0 \
  --edu_threshold 0.8 \
  --percentile_threshold 0.2 \
  --output_file ./output/evolved_instructions.json
```

### Component-Level Usage

```python
from adaptive_evolve import InstructionGenerator, InstructionEvolver, ResponseGenerator, QualityAssessor

# Initialize components
generator = InstructionGenerator(model="meta-llama/Llama-3.2-3B-Instruct")
evolver = InstructionEvolver(evolution_ratio=2.0)
response_gen = ResponseGenerator(model="meta-llama/Llama-3.2-3B-Instruct")
quality = QualityAssessor(edu_threshold=0.8, percentile_threshold=0.2)

# Generate base instructions
base_instructions = generator.generate(seed_instructions, count=100)

# Evolve instructions
evolved_instructions = evolver.evolve(base_instructions)

# Generate responses
instruction_response_pairs = response_gen.generate_responses(evolved_instructions)

# Filter by quality
final_dataset = quality.filter(instruction_response_pairs)
```

## Pipeline Components

### 1. Seed Instruction Loading

Supports two complementary approaches:
- Direct seed instructions covering diverse topics and tasks
- Persona-based instructions generated from persona descriptions

### 2. Instruction Generation

Implements the SelfInstruct approach to generate diverse base instructions from seed prompts, with customizable generation parameters.

### 3. Instruction Evolution

Transforms base instructions using various strategies:
- **Specific**: Adds constraints and specificity to instructions
- **Complex**: Increases the complexity and depth of instructions
- **Detailed**: Adds requirements for detailed responses

Evolution is guided by a complexity scoring function:
```
Complexity(E, I) = α·|E|/|I| + β·TTR(E)/TTR(I) + γ·Spec(E, I)
```
where `|E|` and `|I|` are token counts, `TTR` is Type-Token Ratio, and `Spec` is specificity score.

### 4. Response Generation

Creates high-quality responses for each evolved instruction using Llama-3.2-3B-Instruct.

### 5. Adaptive Quality Assessment

Two-stage filtering process:
- Semantic deduplication using SemHash to remove redundant instructions
- Educational value scoring to retain only high-quality instructions

## Metrics & Evaluation

The pipeline outputs various metrics:
- Token counts and length statistics
- Type-Token Ratio (TTR) for lexical diversity
- Educational value scores and thresholds
- Quality qualification rates


## Citation

If you use AdaptiveEvolve in your research, please cite:

```bibtex
@article{kattamuri2025adaptiveevolve,
  title={AdaptiveEvolve: A Novel Approach to Synthetic Instruction Dataset Generation with Adaptive Quality Assessment},
  author={Kattamuri, Ashish},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
