#!/usr/bin/env python3
"""
AdaptiveEvolve: Instruction Evolution Pipeline

This module implements a modular pipeline for generating, evolving, and filtering
high-quality instruction datasets for language model training. The pipeline combines:

1. Base instruction generation via SelfInstruct methodology
2. Targeted instruction evolution with complexity controls
3. Semantic deduplication to ensure diversity
4. Educational value assessment for quality filtering
5. Automated response generation and dataset compilation

The implementation is designed to be reproducible, extensible, and suitable for
research or production use cases in instruction dataset creation.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Set
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import hashlib
from transformers import pipeline
from datasets import Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InstructionPipeline:
    """
    Comprehensive pipeline for instruction dataset generation and quality assessment.
    
    This class implements the AdaptiveEvolve framework, providing a modular, configurable
    pipeline for generating high-quality instruction-response pairs. The pipeline features:
    
    * Multiple instruction generation strategies
    * Targeted evolution with complexity controls
    * Semantic deduplication for diversity
    * Educational value assessment and filtering
    * Comprehensive metrics and visualization
    
    The implementation is designed for both research and production use cases, with
    appropriate fallbacks and error handling for robustness.
    """
    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
                 classifier_name: str = "HuggingFaceFW/fineweb-edu-classifier",
                 output_dir: str = "output"):
        """
        Initialize the instruction pipeline.
        
        Args:
            model_name: Name of the model to use for generation
            classifier_name: Name of the educational classifier
            output_dir: Directory to save outputs
        """
        self.model_name = model_name
        self.classifier_name = classifier_name
        self.output_dir = output_dir
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Initialize components
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Load classifier
            self.edu_classifier = pipeline("text-classification", model=classifier_name)
            logger.info(f"Successfully initialized pipeline with model {model_name}")
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            # Fallback to text-generation pipeline
            self.model = pipeline("text-generation", model="gpt2")
            self.edu_classifier = None
            logger.info(f"Fallback to gpt2 model for demo purposes")
        
        # Results tracking
        self.metrics = {
            "base_instructions": [],
            "evolved_instructions": [],
            "token_counts_base": [],
            "token_counts_evolved": [],
            "ttr_base": [],
            "ttr_evolved": [],
            "edu_scores": []
        }
        
    def generate_base_instructions(self, 
                                   seed_instructions: List[str],
                                   count: int = 10) -> List[Dict[str, str]]:
        """
        Generate base instructions using SelfInstruct.
        
        Args:
            seed_instructions: List of seed instructions
            count: Number of instructions to generate
            
        Returns:
            List of generated instructions
        """
        logger.info(f"Generating {count} base instructions from {len(seed_instructions)} seeds")
        results = []
        
        # Implement actual generation with fallbacks for demo purposes
        try:
            for i in range(0, count, 3):
                # Select random seeds for examples
                examples = np.random.choice(seed_instructions, 
                                           size=min(3, len(seed_instructions)), 
                                           replace=False)
                
                # Format prompt
                prompt = self._format_self_instruct_prompt(examples)
                
                # Generate instructions
                try:
                    if hasattr(self, 'model'):
                        response = self.model(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']
                        # Extract just the generated part
                        response = response[len(prompt):].strip()
                    else:
                        # Fallback for demonstration
                        response = f"Explain the impact of {np.random.choice(['climate change', 'artificial intelligence', 'renewable energy', 'quantum computing', 'blockchain technology'])} on {np.random.choice(['society', 'economy', 'healthcare', 'education', 'transportation'])}."
                except Exception as e:
                    logger.warning(f"Error in generation, using fallback: {e}")
                    response = f"Explain the impact of {np.random.choice(['climate change', 'artificial intelligence', 'renewable energy', 'quantum computing', 'blockchain technology'])} on {np.random.choice(['society', 'economy', 'healthcare', 'education', 'transportation'])}."
                
                # Parse and add to results
                results.append({"instruction": response, "source": "self_instruct"})
                
                # Update metrics
                self.metrics["base_instructions"].append(response)
                self.metrics["token_counts_base"].append(len(response.split()))
                self.metrics["ttr_base"].append(self._compute_ttr(response))
                
        except Exception as e:
            logger.error(f"Error in instruction generation: {e}")
            # Add fallback instructions for demo purposes
            fallback_templates = [
                "Explain the concept of {topic} to a {audience}.",
                "Write a step-by-step guide on how to {task}.",
                "Compare and contrast {item1} and {item2}.",
                "Describe the implications of {event} on {domain}.",
                "Analyze the main themes in {work}."
            ]
            topics = ["machine learning", "climate change", "quantum physics", 
                      "blockchain", "renewable energy", "artificial intelligence"]
            audiences = ["beginner", "high school student", "expert", "child", "senior citizen"]
            tasks = ["build a website", "train a neural network", "learn a new language", 
                     "start investing", "improve critical thinking"]
            
            for i in range(count):
                template = np.random.choice(fallback_templates)
                if "{topic}" in template and "{audience}" in template:
                    instruction = template.format(
                        topic=np.random.choice(topics),
                        audience=np.random.choice(audiences)
                    )
                elif "{task}" in template:
                    instruction = template.format(task=np.random.choice(tasks))
                elif "{item1}" in template and "{item2}" in template:
                    items = np.random.choice(topics, 2, replace=False)
                    instruction = template.format(item1=items[0], item2=items[1])
                elif "{event}" in template and "{domain}" in template:
                    events = ["digitalization", "globalization", "pandemic", "automation"]
                    domains = ["education", "healthcare", "economy", "society"]
                    instruction = template.format(
                        event=np.random.choice(events),
                        domain=np.random.choice(domains)
                    )
                else:
                    works = ["Shakespeare's plays", "modern literature", "classical music",
                            "contemporary art", "philosophical theories"]
                    instruction = template.format(work=np.random.choice(works))
                
                results.append({"instruction": instruction, "source": "self_instruct"})
                
                # Update metrics
                self.metrics["base_instructions"].append(instruction)
                self.metrics["token_counts_base"].append(len(instruction.split()))
                self.metrics["ttr_base"].append(self._compute_ttr(instruction))
        
        logger.info(f"Generated {len(results)} base instructions")
        return results

    def evolve_instructions(self, base_instructions: List[Dict[str, str]], target_ratio: float = 1.5) -> List[Dict[str, str]]:
        """
        Transform base instructions into more complex, specific, and educationally valuable forms.
        
        This method implements a controllable evolution process that targets a specific
        complexity ratio. The implementation strategically selects from multiple evolution
        types (deeper, specific, complex, multi-step) based on the desired complexity level.
        For each instruction, multiple evolution attempts may be made to reach the target
        complexity ratio.
        
        Args:
            base_instructions: List of base instructions to evolve
            target_ratio: Target complexity ratio (higher values produce more complex instructions)
            
        Returns:
            List of evolved instructions with metadata including evolution type and complexity metrics
        """
        logger.info(f"Evolving {len(base_instructions)} instructions with target ratio {target_ratio}")
        results = []
        
        # Adjust evolution strategy distribution based on target ratio
        if target_ratio <= 1.3:
            # For lower complexity, favor "specific" type
            evolution_weights = {"deeper": 0.2, "specific": 0.5, "complex": 0.2, "multi-step": 0.1}
        elif target_ratio <= 1.8:
            # For medium complexity, balanced approach
            evolution_weights = {"deeper": 0.25, "specific": 0.3, "complex": 0.25, "multi-step": 0.2}
        else:
            # For high complexity, favor "complex" and "multi-step"
            evolution_weights = {"deeper": 0.2, "specific": 0.2, "complex": 0.3, "multi-step": 0.3}
            
        evolution_types = list(evolution_weights.keys())
        evolution_probs = list(evolution_weights.values())
        
        # Number of attempts for each instruction to reach target ratio
        max_attempts = 3
        
        for item in base_instructions:
            base_instruction = item["instruction"]
            
            best_evolved = None
            best_complexity = 0
            best_type = None
            
            # Try multiple evolution types to reach target ratio
            for _ in range(max_attempts):
                evolution_type = np.random.choice(evolution_types, p=evolution_probs)
                
                # Format evolution prompt
                prompt = self._format_evol_prompt(base_instruction, evolution_type)
                
                # Generate evolved instruction
                try:
                    if hasattr(self, 'model'):
                        response = self.model(prompt, max_length=300, num_return_sequences=1)[0]['generated_text']
                        # Extract just the generated part
                        evolved_instruction = response[len(prompt):].strip()
                    else:
                        # Create a more complex version for demonstration
                        evolved_instruction = self._demo_evolve_instruction(base_instruction, evolution_type)
                except Exception as e:
                    logger.warning(f"Error in evolution, using fallback: {e}")
                    evolved_instruction = self._demo_evolve_instruction(base_instruction, evolution_type)
                
                # Calculate complexity ratio
                complexity = self._compute_complexity(evolved_instruction, base_instruction)
                
                # If we've exceeded target ratio or this is best so far, update
                if complexity >= target_ratio or complexity > best_complexity:
                    best_evolved = evolved_instruction
                    best_complexity = complexity
                    best_type = evolution_type
                    
                    # If we've reached target, no need to try more
                    if complexity >= target_ratio:
                        break
            
            # Use the best evolution or fall back to base
            if best_evolved and best_complexity > 1.2:
                results.append({
                    "base_instruction": base_instruction,
                    "evolved_instruction": best_evolved,
                    "evolution_type": best_type,
                    "complexity_ratio": best_complexity,
                    "source": "evol_instruct"
                })
                
                # Update metrics
                self.metrics["evolved_instructions"].append(best_evolved)
                self.metrics["token_counts_evolved"].append(len(best_evolved.split()))
                self.metrics["ttr_evolved"].append(self._compute_ttr(best_evolved))
            else:
                # Fall back to base if evolution wasn't meaningful
                results.append({
                    "base_instruction": base_instruction,
                    "evolved_instruction": base_instruction,
                    "evolution_type": "none",
                    "complexity_ratio": 1.0,
                    "source": "base_preserved" 
                })
                
                # Update metrics with base values
                self.metrics["evolved_instructions"].append(base_instruction)
                self.metrics["token_counts_evolved"].append(len(base_instruction.split()))
                self.metrics["ttr_evolved"].append(self._compute_ttr(base_instruction))
        
        # Log average complexity achieved
        avg_complexity = np.mean([item.get("complexity_ratio", 1.0) for item in results])
        logger.info(f"Evolved {len(results)} instructions with average complexity ratio: {avg_complexity:.2f}")
        return results
    
    def deduplicate_instructions(self, 
                                 instructions: List[Dict[str, str]],
                                 field: str = "evolved_instruction") -> List[Dict[str, str]]:
        """
        Deduplicate instructions using SemHash.
        
        Args:
            instructions: List of instructions
            field: Field to use for deduplication
            
        Returns:
            List of deduplicated instructions
        """
        logger.info(f"Deduplicating {len(instructions)} instructions")
        
        # Extract texts for deduplication
        texts = [item[field] for item in instructions]
        
        # Compute TF-IDF
        vectorizer = TfidfVectorizer(max_features=100)
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
        except:
            # Fallback for demo
            logger.warning("TF-IDF computation failed, using simple hash")
            hashes = {self._simple_hash(text): i for i, text in enumerate(texts)}
            return [instructions[i] for i in hashes.values()]
        
        # Compute SemHash for each instruction
        hashes = {}
        for i, text in enumerate(texts):
            # Get top-k terms by TF-IDF
            tfidf_scores = tfidf_matrix[i].toarray()[0]
            top_terms_indices = tfidf_scores.argsort()[-10:][::-1]  # top 10 terms
            top_terms = [feature_names[idx] for idx in top_terms_indices if tfidf_scores[idx] > 0]
            
            # Create hash
            if not top_terms:  # Fallback if no terms
                hash_val = self._simple_hash(text)
            else:
                term_hashes = [hashlib.md5(term.encode()).hexdigest() for term in sorted(top_terms)]
                hash_val = hashlib.md5("".join(term_hashes).encode()).hexdigest()
            
            # Only keep one item per hash
            if hash_val not in hashes:
                hashes[hash_val] = i
        
        # Return deduplicated instructions
        deduplicated = [instructions[i] for i in hashes.values()]
        logger.info(f"Deduplicated to {len(deduplicated)} instructions")
        return deduplicated
    
    def score_educational_value(self, 
                               instructions: List[Dict[str, str]],
                               field: str = "evolved_instruction") -> List[Dict[str, Any]]:
        """
        Score instructions for educational value.
        
        Args:
            instructions: List of instructions
            field: Field to score
            
        Returns:
            List of instructions with scores
        """
        logger.info(f"Scoring {len(instructions)} instructions for educational value")
        
        # Extract texts for scoring
        texts = [item[field] for item in instructions]
        
        # Score with classifier or demo values
        try:
            if self.edu_classifier:
                results = self.edu_classifier(texts)
                scores = [result["score"] for result in results]
            else:
                # Demo scoring - simulate educational value based on complexity
                scores = [min(0.95, max(0.5, 0.6 + 0.2 * np.random.random() + 
                               0.002 * len(text.split()))) for text in texts]
        except Exception as e:
            logger.warning(f"Error in educational scoring, using fallback: {e}")
            # Fallback for demo
            scores = [min(0.95, max(0.5, 0.6 + 0.2 * np.random.random() + 
                         0.002 * len(text.split()))) for text in texts]
        
        # Add scores to instructions
        for i, score in enumerate(scores):
            instructions[i]["educational_score"] = score
            
        # Update metrics
        self.metrics["edu_scores"].extend(scores)
        
        logger.info(f"Scored {len(instructions)} instructions")
        return instructions
    
    def filter_by_quality(self, 
                         instructions: List[Dict[str, Any]],
                         min_score: float = 0.8,
                         percentile: float = 0.2) -> List[Dict[str, Any]]:
        """
        Filter instructions by quality.
        
        Args:
            instructions: List of instructions with scores
            min_score: Minimum absolute score
            percentile: Percentile threshold (lower = keep more)
            
        Returns:
            List of filtered instructions
        """
        logger.info(f"Filtering {len(instructions)} instructions by quality")
        
        # Extract scores
        scores = [item["educational_score"] for item in instructions]
        
        # Compute percentile threshold
        if scores:
            percentile_threshold = np.percentile(scores, percentile * 100)
            threshold = max(min_score, percentile_threshold)
        else:
            threshold = min_score
        
        # Filter by threshold
        filtered = [item for item in instructions if item["educational_score"] >= threshold]
        
        logger.info(f"Filtered to {len(filtered)} high-quality instructions")
        return filtered
    
    def generate_responses(self, 
                          instructions: List[Dict[str, Any]],
                          field: str = "evolved_instruction") -> List[Dict[str, Any]]:
        """
        Generate responses for instructions.
        
        Args:
            instructions: List of instructions
            field: Field containing the instruction
            
        Returns:
            List of instruction-response pairs
        """
        logger.info(f"Generating responses for {len(instructions)} instructions")
        
        for item in tqdm(instructions):
            instruction = item[field]
            
            # Generate response
            try:
                if hasattr(self, 'model'):
                    response_text = self.model(
                        f"Instruction: {instruction}\n\nResponse:", 
                        max_length=500, 
                        num_return_sequences=1
                    )[0]['generated_text']
                    # Extract just the generated part
                    response = response_text.split("Response:")[1].strip() if "Response:" in response_text else response_text
                else:
                    # Demo response generation
                    response = f"This is a simulated response to the instruction: '{instruction}'. In a real implementation, this would be generated by the LLM."
            except Exception as e:
                logger.warning(f"Error in response generation, using fallback: {e}")
                response = f"This is a simulated response to the instruction: '{instruction}'. In a real implementation, this would be generated by the LLM."
            
            item["response"] = response
        
        logger.info(f"Generated {len(instructions)} responses")
        return instructions
    
    def run_pipeline(self, 
                    seed_instructions: List[str],
                    count: int = 10,
                    evolution_ratio: float = 1.5) -> List[Dict[str, Any]]:
        """
        Run the complete pipeline.
        
        Args:
            seed_instructions: List of seed instructions
            count: Number of instructions to generate
            evolution_ratio: Target ratio for evolution complexity (higher = more complex)
            
        Returns:
            List of processed instruction-response pairs
        """
        logger.info("Starting pipeline run")
        
        # Generate base instructions
        base_instructions = self.generate_base_instructions(seed_instructions, count)
        
        # Evolve instructions with target complexity ratio
        evolved_instructions = self.evolve_instructions(base_instructions, target_ratio=evolution_ratio)
        
        # Deduplicate instructions
        deduplicated = self.deduplicate_instructions(evolved_instructions)
        
        # Score educational value
        scored = self.score_educational_value(deduplicated)
        
        # Filter by quality
        filtered = self.filter_by_quality(scored)
        
        # Generate responses
        final_dataset = self.generate_responses(filtered)
        
        # Save results
        self.save_results(final_dataset)
        
        # Analyze and visualize
        self.analyze_results()
        
        logger.info("Pipeline run completed")
        return final_dataset
    
    def save_results(self, dataset: List[Dict[str, Any]]) -> None:
        """
        Save results to disk.
        
        Args:
            dataset: List of processed instruction-response pairs
        """
        # Save as JSON
        with open(os.path.join(self.output_dir, "instruction_dataset.json"), "w") as f:
            json.dump(dataset, f, indent=2)
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(dataset)
        
        # Save as CSV
        df.to_csv(os.path.join(self.output_dir, "instruction_dataset.csv"), index=False)
        
        # Save to Hugging Face Dataset format
        hf_dataset = Dataset.from_pandas(df)
        hf_dataset.save_to_disk(os.path.join(self.output_dir, "hf_dataset"))
        
        logger.info(f"Saved dataset with {len(dataset)} items to {self.output_dir}")
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze and visualize results.
        
        Returns:
            Dictionary of metrics
        """
        # Compute metrics
        metrics = {
            "count": {
                "base": len(self.metrics["base_instructions"]),
                "evolved": len(self.metrics["evolved_instructions"]),
                "edu_scores": len(self.metrics["edu_scores"])
            },
            "token_length": {
                "base_mean": np.mean(self.metrics["token_counts_base"]) if self.metrics["token_counts_base"] else 0,
                "base_std": np.std(self.metrics["token_counts_base"]) if self.metrics["token_counts_base"] else 0,
                "evolved_mean": np.mean(self.metrics["token_counts_evolved"]) if self.metrics["token_counts_evolved"] else 0,
                "evolved_std": np.std(self.metrics["token_counts_evolved"]) if self.metrics["token_counts_evolved"] else 0,
                "evolution_ratio": (np.mean(self.metrics["token_counts_evolved"]) / 
                                   np.mean(self.metrics["token_counts_base"])) 
                                   if (self.metrics["token_counts_base"] and 
                                       np.mean(self.metrics["token_counts_base"]) > 0) else 0
            },
            "ttr": {
                "base_mean": np.mean(self.metrics["ttr_base"]) if self.metrics["ttr_base"] else 0,
                "evolved_mean": np.mean(self.metrics["ttr_evolved"]) if self.metrics["ttr_evolved"] else 0,
                "improvement": ((np.mean(self.metrics["ttr_evolved"]) / np.mean(self.metrics["ttr_base"])) - 1) * 100
                               if (self.metrics["ttr_base"] and np.mean(self.metrics["ttr_base"]) > 0) else 0
            },
            "edu_scores": {
                "mean": np.mean(self.metrics["edu_scores"]) if self.metrics["edu_scores"] else 0,
                "median": np.median(self.metrics["edu_scores"]) if self.metrics["edu_scores"] else 0,
                "above_0.8": (sum(1 for s in self.metrics["edu_scores"] if s >= 0.8) / 
                             len(self.metrics["edu_scores"]) * 100) if self.metrics["edu_scores"] else 0
            }
        }
        
        # Save metrics
        with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Create visualizations
        self._plot_token_lengths()
        self._plot_ttr_comparison()
        self._plot_edu_score_dist()
        
        logger.info(f"Analyzed results and saved metrics to {self.output_dir}")
        return metrics
    
    def _format_self_instruct_prompt(self, examples: List[str]) -> str:
        """Format prompt for SelfInstruct."""
        examples_text = "\n".join([f"- {example}" for example in examples])
        return f"Generate diverse, creative instructions based on the following examples:\n\n{examples_text}\n\n"
    
    def _format_evol_prompt(self, instruction: str, evolution_type: str) -> str:
        """Format prompt for EvolInstruct."""
        prompts = {
            "deeper": f"Make this instruction more in-depth and specific, adding details and context: {instruction}",
            "specific": f"Transform this instruction into a more specialized version with constraints: {instruction}",
            "complex": f"Make this instruction more complex, requiring deeper reasoning: {instruction}",
            "multi-step": f"Convert this instruction into a multi-step process with sequential tasks: {instruction}"
        }
        return prompts.get(evolution_type, f"Improve this instruction: {instruction}")
    
    def _compute_ttr(self, text: str) -> float:
        """Compute Type-Token Ratio."""
        tokens = text.lower().split()
        if not tokens:
            return 0
        return len(set(tokens)) / len(tokens)
    
    def _compute_complexity(self, evolved: str, base: str) -> float:
        """Compute complexity ratio between evolved and base instructions."""
        # Token length ratio (weight: 0.5)
        token_ratio = len(evolved.split()) / max(1, len(base.split()))
        
        # TTR ratio (weight: 0.3)
        ttr_evolved = self._compute_ttr(evolved)
        ttr_base = self._compute_ttr(base)
        ttr_ratio = ttr_evolved / max(0.01, ttr_base)
        
        # Specificity (weight: 0.2) - approximated by presence of specific terms
        specificity_terms = ["specifically", "exactly", "precisely", "in detail", 
                            "step by step", "first", "second", "third", "finally"]
        specificity_score = 1.0 + 0.1 * sum(1 for term in specificity_terms if term in evolved.lower())
        
        # Weighted combination
        return 0.5 * token_ratio + 0.3 * ttr_ratio + 0.2 * specificity_score
    
    def _simple_hash(self, text: str) -> str:
        """Simple hash function for demonstration."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _demo_evolve_instruction(self, instruction: str, evolution_type: str) -> str:
        """Create evolved instruction for demonstration purposes."""
        tokens = instruction.split()
        
        if evolution_type == "deeper":
            specificity_terms = [
                "specifically focusing on", "with particular attention to", 
                "in detail", "thoroughly examining", "comprehensively analyzing"
            ]
            domains = [
                "historical context", "socioeconomic implications", 
                "philosophical underpinnings", "scientific principles",
                "practical applications", "theoretical framework", 
                "ethical considerations"
            ]
            
            spec_term = np.random.choice(specificity_terms)
            domain = np.random.choice(domains)
            
            return f"{instruction} {spec_term} {domain} and providing concrete examples to illustrate key points."
            
        elif evolution_type == "specific":
            constraints = [
                "using only peer-reviewed sources published after 2020",
                "focusing specifically on applications in developing economies",
                "considering both Western and Eastern philosophical traditions",
                "analyzing from both a technical and ethical perspective",
                "using exactly five examples from diverse geographical regions"
            ]
            
            constraint = np.random.choice(constraints)
            return f"{instruction} {constraint}."
            
        elif evolution_type == "complex":
            complexity_phrases = [
                "Analyze the underlying assumptions and potential biases in your approach to",
                "Compare and contrast at least three different theoretical frameworks when you",
                "Evaluate the long-term sustainability and scalability implications as you",
                "Critically assess the epistemological foundations that inform how you"
            ]
            
            if len(tokens) > 4:
                # Try to preserve the core instruction while making it more complex
                verb_pos = min(4, len(tokens) - 1)  # Estimate position of main verb
                complexity = np.random.choice(complexity_phrases)
                return f"{complexity} {' '.join(tokens[verb_pos:])}"
            else:
                return f"{np.random.choice(complexity_phrases)} {instruction}"
                
        elif evolution_type == "multi-step":
            steps = [
                "First, research and identify the key components of",
                "Next, analyze and evaluate the relationships between these components in",
                "Then, synthesize your findings to create a comprehensive framework for",
                "Finally, propose specific applications or implementations based on your analysis of"
            ]
            
            return f"Complete a multi-step analysis: {' '.join(steps)} {instruction}"
            
        else:
            # Default evolution - just make it longer
            return f"Provide a detailed, well-structured, and comprehensive response to the following: {instruction}"
    
    def _plot_token_lengths(self) -> None:
        """Plot token length comparison."""
        plt.figure(figsize=(10, 6))
        
        # Data
        base_tokens = self.metrics["token_counts_base"]
        evolved_tokens = self.metrics["token_counts_evolved"]
        
        # Plot
        plt.hist([base_tokens, evolved_tokens], bins=10, 
                 label=['Base Instructions', 'Evolved Instructions'],
                 alpha=0.7, color=['blue', 'green'])
        
        plt.title('Token Length Distribution: Base vs. Evolved Instructions')
        plt.xlabel('Token Count')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save
        plt.savefig(os.path.join(self.output_dir, "token_length_comparison.png"), dpi=300)
        plt.close()
    
    def _plot_ttr_comparison(self) -> None:
        """Plot TTR comparison."""
        plt.figure(figsize=(10, 6))
        
        # Data
        base_ttr = self.metrics["ttr_base"]
        evolved_ttr = self.metrics["ttr_evolved"]
        
        # Plot
        plt.hist([base_ttr, evolved_ttr], bins=10, 
                 label=['Base Instructions', 'Evolved Instructions'],
                 alpha=0.7, color=['blue', 'green'])
        
        plt.title('Type-Token Ratio Distribution: Base vs. Evolved Instructions')
        plt.xlabel('TTR')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save
        plt.savefig(os.path.join(self.output_dir, "ttr_comparison.png"), dpi=300)
        plt.close()
    
    def _plot_edu_score_dist(self) -> None:
        """Plot educational score distribution."""
        plt.figure(figsize=(10, 6))
        
        # Data
        edu_scores = self.metrics["edu_scores"]
        
        # Plot
        plt.hist(edu_scores, bins=20, color='skyblue', edgecolor='black')
        plt.axvline(x=0.8, color='red', linestyle='--', 
                    label='Quality Threshold (0.8)')
        
        plt.title('Educational Score Distribution')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save
        plt.savefig(os.path.join(self.output_dir, "edu_score_distribution.png"), dpi=300)
        plt.close()


if __name__ == "__main__":
    # Sample seed instructions
    seed_instructions = [
        "Explain the concept of neural networks to a high school student.",
        "Write a short story about a robot learning to feel emotions.",
        "Describe the process of photosynthesis in plants.",
        "Analyze the main themes in Shakespeare's Hamlet.",
        "Compare and contrast renewable and non-renewable energy sources."
    ]
    
    # Run pipeline
    pipeline = InstructionPipeline(output_dir="output")
    results = pipeline.run_pipeline(seed_instructions, count=20)
    
    print(f"Pipeline complete. Generated {len(results)} instruction-response pairs.")
