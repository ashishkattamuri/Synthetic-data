#!/usr/bin/env python3
"""
Persona-based instruction generation.

This module provides utilities to generate contextualized instructions based on user personas.
The persona-based approach creates more realistic and diverse instructions by considering
different user backgrounds, interests, and knowledge domains.
"""

import os
import json
import logging
import random
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Example personas from the dataset
SAMPLE_PERSONAS = [
    {
        "name": "Dr. Maya Thompson",
        "age": 42,
        "occupation": "Environmental Scientist",
        "interests": ["climate research", "sustainable living", "hiking"],
        "background": "PhD in Environmental Science, researching climate change impacts"
    },
    {
        "name": "Alex Rivera",
        "age": 28,
        "occupation": "Software Developer",
        "interests": ["machine learning", "open source", "gaming"],
        "background": "Computer Science degree, working on AI applications"
    },
    {
        "name": "Sarah Chen",
        "age": 35,
        "occupation": "High School Teacher",
        "interests": ["education technology", "literature", "travel"],
        "background": "Masters in Education, 10 years teaching experience"
    },
    {
        "name": "Marcus Johnson",
        "age": 52,
        "occupation": "Financial Analyst",
        "interests": ["investment strategies", "economics", "jazz music"],
        "background": "MBA, 25 years in finance industry"
    },
    {
        "name": "Olivia Patel",
        "age": 31,
        "occupation": "Healthcare Worker",
        "interests": ["medical research", "fitness", "cooking"],
        "background": "Nursing degree, specialized in emergency care"
    }
]

# Templates for converting personas to instructions
INSTRUCTION_TEMPLATES = [
    "As {name}, a {age}-year-old {occupation} interested in {interest}, write a detailed explanation of {topic}.",
    "Taking on the perspective of {name} with a background in {background}, provide insights on {topic}.",
    "You are {name}, who works as a {occupation}. Write a guide about {topic} that incorporates your expertise.",
    "As {name} with interests in {interest}, create a comprehensive tutorial on {topic}.",
    "From the viewpoint of {name}, an experienced {occupation}, answer this question in detail: {question}",
    "Imagine you are {name} with expertise in {background}. How would you explain {topic} to someone new to the field?",
    "As {name}, write a persuasive argument about {topic} that draws on your background in {background}.",
    "Taking on the role of {name}, create a step-by-step guide for {task} that incorporates your knowledge as a {occupation}."
]

TOPICS = [
    "sustainable energy solutions",
    "artificial intelligence ethics",
    "effective teaching strategies",
    "investment portfolio diversification",
    "healthcare innovation",
    "climate change mitigation",
    "software development best practices",
    "financial literacy education",
    "educational technology integration",
    "work-life balance strategies",
    "data privacy concerns",
    "mental health awareness",
    "digital transformation in business",
    "inclusive classroom techniques",
    "remote work productivity"
]

QUESTIONS = [
    "What are the most promising approaches to combat climate change?",
    "How can artificial intelligence be used ethically in decision-making?",
    "What teaching methods are most effective for engaging diverse learners?",
    "How should individuals prepare financially for economic uncertainty?",
    "What innovations will transform healthcare in the next decade?",
    "How can organizations effectively implement digital transformation?",
    "What skills will be most valuable in the future job market?",
    "How can we balance technological advancement with privacy concerns?"
]

TASKS = [
    "implementing a machine learning model",
    "creating an environmentally sustainable home",
    "developing an effective lesson plan",
    "building a diversified investment portfolio",
    "improving healthcare outcomes",
    "reducing your carbon footprint",
    "learning a programming language",
    "teaching complex concepts to beginners"
]

def generate_persona_instructions(count: int = 50) -> List[str]:
    """
    Generate instructions based on personas, similar to the dvilasuero/finepersonas-v0.1-tiny approach.
    
    Args:
        count: Number of instructions to generate
        
    Returns:
        List of generated instructions
    """
    instructions = []
    
    # In a real implementation, this would fetch from the actual dataset
    # For demonstration, we'll use our sample personas
    personas = SAMPLE_PERSONAS
    
    # Generate instructions
    for _ in range(count):
        # Select random persona and template
        persona = random.choice(personas)
        template = random.choice(INSTRUCTION_TEMPLATES)
        
        # Fill in template
        if "{interest}" in template:
            interest = random.choice(persona["interests"])
        else:
            interest = None
            
        if "{topic}" in template:
            topic = random.choice(TOPICS)
        else:
            topic = None
            
        if "{question}" in template:
            question = random.choice(QUESTIONS)
        else:
            question = None
            
        if "{task}" in template:
            task = random.choice(TASKS)
        else:
            task = None
        
        # Create instruction
        instruction = template.format(
            name=persona["name"],
            age=persona["age"],
            occupation=persona["occupation"],
            background=persona["background"],
            interest=interest if interest else random.choice(persona["interests"]),
            topic=topic if topic else random.choice(TOPICS),
            question=question if question else random.choice(QUESTIONS),
            task=task if task else random.choice(TASKS)
        )
        
        instructions.append(instruction)
    
    return instructions

def save_persona_instructions(count: int = 50, output_file: str = "persona_instructions.json"):
    """
    Generate and save persona-based instructions to a file.
    
    Args:
        count: Number of instructions to generate
        output_file: Path to save the instructions
    """
    instructions = generate_persona_instructions(count)
    
    with open(output_file, "w") as f:
        json.dump(instructions, f, indent=2)
    
    logger.info(f"Generated {len(instructions)} persona-based instructions saved to {output_file}")
    return instructions

if __name__ == "__main__":
    # Generate and save persona instructions
    save_persona_instructions(50, "persona_instructions.json")
