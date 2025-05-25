#!/usr/bin/env python3
"""
Benchmark Dataset Utilities for AdaptiveEvolve

This module provides utilities for sampling and processing instructions from
standard language model instruction datasets for comparative evaluation purposes.
The implementation includes realistic simulations of the FLAN, Alpaca, and T0
datasets with comparable structure and linguistic characteristics.

These utilities enable rigorous comparison between AdaptiveEvolve-generated
instructions and established benchmarks across metrics such as length,
lexical diversity, educational value, and quality threshold rates.
"""

import os
import json
import random
import logging
from typing import List
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sample sizes for benchmark datasets
SAMPLE_SIZE = 100

def get_flan_sample() -> List[str]:
    """
    Generate a representative sample of FLAN-style instructions.
    
    This function creates instruction examples that match the structure, complexity,
    and domain coverage of the FLAN (Fine-tuned LAnguage Net) dataset, focusing on
    its characteristic task-oriented, templated instruction patterns.
    
    Returns:
        List of instruction strings formatted in FLAN style
    """
    # In a real implementation, this would load from the actual FLAN dataset
    # For demonstration, we'll simulate with realistic examples
    
    flan_templates = [
        "Translate the following English text to {language}: {text}",
        "Write a {length} summary of the following text: {text}",
        "Answer the following multiple-choice question: {question}",
        "Generate a list of {count} {topic}",
        "Explain the concept of {concept} to a {audience}",
        "What is the sentiment of this review: {review}",
        "Write a {tone} email about {topic}",
        "Classify the following text into one of these categories: {categories}. Text: {text}",
        "Generate a creative {creative_type} about {topic}",
        "Answer this question step by step: {question}"
    ]
    
    languages = ["French", "Spanish", "German", "Chinese", "Russian", "Japanese", "Italian"]
    lengths = ["short", "concise", "detailed", "comprehensive", "brief"]
    audiences = ["5-year-old", "high school student", "college freshman", "expert", "beginner", "non-technical person"]
    tones = ["formal", "informal", "friendly", "professional", "enthusiastic", "urgent"]
    creative_types = ["poem", "story", "advertisement", "slogan", "song", "joke"]
    topics = ["artificial intelligence", "climate change", "renewable energy", "quantum physics", 
              "psychology", "education", "healthcare", "transportation", "space exploration"]
    concepts = ["machine learning", "relativity", "photosynthesis", "supply and demand", 
                "democracy", "genetic engineering", "blockchain", "natural selection"]
    
    instructions = []
    for _ in range(SAMPLE_SIZE):
        template = random.choice(flan_templates)
        
        if "{language}" in template:
            instruction = template.format(
                language=random.choice(languages),
                text="The quick brown fox jumps over the lazy dog."
            )
        elif "{length}" in template:
            instruction = template.format(
                length=random.choice(lengths),
                text="[Long article about recent scientific discoveries]"
            )
        elif "multiple-choice" in template:
            instruction = template.format(
                question="What is the capital of France? A) London B) Berlin C) Paris D) Madrid"
            )
        elif "{count}" in template:
            instruction = template.format(
                count=random.randint(3, 10),
                topic=random.choice(topics)
            )
        elif "{concept}" in template:
            instruction = template.format(
                concept=random.choice(concepts),
                audience=random.choice(audiences)
            )
        elif "sentiment" in template:
            instruction = template.format(
                review="I really enjoyed this product. It exceeded my expectations in every way."
            )
        elif "{tone}" in template:
            instruction = template.format(
                tone=random.choice(tones),
                topic=random.choice(topics)
            )
        elif "Classify" in template:
            instruction = template.format(
                categories="Business, Technology, Entertainment, Sports",
                text="Apple announced its newest iPhone model yesterday at their annual conference."
            )
        elif "creative" in template:
            instruction = template.format(
                creative_type=random.choice(creative_types),
                topic=random.choice(topics)
            )
        elif "step by step" in template:
            instruction = template.format(
                question="How does a nuclear reactor generate electricity?"
            )
        
        instructions.append(instruction)
    
    return instructions

def get_alpaca_sample() -> List[str]:
    """
    Generate a representative sample of Alpaca-style instructions.
    
    This function creates instruction examples that match the structure and characteristics
    of the Alpaca dataset, focusing on its more concise, direct instruction format and
    covering similar task domains as the original Stanford Alpaca dataset.
    
    Returns:
        List of instruction strings formatted in Alpaca style
    """
    # In a real implementation, this would load from the actual Alpaca dataset
    # For demonstration, we'll simulate with realistic examples
    
    alpaca_templates = [
        "Write a blog post about {topic}",
        "Tell me about {subject}",
        "Explain how {process} works",
        "What are the pros and cons of {topic}?",
        "Write a cover letter for a {job_position} position",
        "Give me tips on how to improve my {skill}",
        "What's the difference between {thing1} and {thing2}?",
        "How do I {task}?",
        "Write a story that includes the following elements: {elements}",
        "Create a meal plan for someone who wants to {diet_goal}"
    ]
    
    topics = ["working remotely", "sustainable living", "digital privacy", "machine learning applications",
              "mindfulness", "personal finance", "cloud computing", "future of transportation"]
    subjects = ["the Renaissance period", "quantum computing", "cognitive biases", "the history of the internet",
                "cryptocurrency", "climate change impacts", "artificial general intelligence"]
    processes = ["photosynthesis", "machine learning", "cellular respiration", "blockchain technology",
                 "DNA replication", "cloud computing", "natural language processing"]
    job_positions = ["software engineer", "data scientist", "marketing manager", "UX designer",
                     "product manager", "financial analyst", "research scientist"]
    skills = ["public speaking", "coding", "writing", "time management", "networking", "negotiation", "leadership"]
    thing_pairs = [("machine learning", "deep learning"), ("coding", "programming"), 
                   ("empathy", "sympathy"), ("AI", "ML"), ("stocks", "bonds")]
    tasks = ["build a simple website", "improve my memory", "learn a new language efficiently",
             "start investing", "train for a marathon", "negotiate a salary increase"]
    diet_goals = ["lose weight", "build muscle", "improve heart health", "increase energy levels",
                  "reduce inflammation", "manage diabetes"]
    
    instructions = []
    for _ in range(SAMPLE_SIZE):
        template = random.choice(alpaca_templates)
        
        if "{topic}" in template:
            instruction = template.format(topic=random.choice(topics))
        elif "{subject}" in template:
            instruction = template.format(subject=random.choice(subjects))
        elif "{process}" in template:
            instruction = template.format(process=random.choice(processes))
        elif "pros and cons" in template:
            instruction = template.format(topic=random.choice(topics))
        elif "cover letter" in template:
            instruction = template.format(job_position=random.choice(job_positions))
        elif "{skill}" in template:
            instruction = template.format(skill=random.choice(skills))
        elif "{thing1}" in template and "{thing2}" in template:
            thing1, thing2 = random.choice(thing_pairs)
            instruction = template.format(thing1=thing1, thing2=thing2)
        elif "{task}" in template:
            instruction = template.format(task=random.choice(tasks))
        elif "story" in template:
            elements = ", ".join(random.sample(["love", "betrayal", "redemption", "adventure", "mystery", "conflict"], 3))
            instruction = template.format(elements=elements)
        elif "meal plan" in template:
            instruction = template.format(diet_goal=random.choice(diet_goals))
            
        instructions.append(instruction)
    
    return instructions

def get_t0_sample() -> List[str]:
    """
    Generate a representative sample of T0-style instructions.
    
    This function creates instruction examples that match the structure and characteristics
    of the T0 (T-zero) dataset, focusing on its prompted task format with diverse natural
    language understanding and reasoning tasks similar to those used in the T0 training mixture.
    
    Returns:
        List of instruction strings formatted in T0 style
    """
    # In a real implementation, this would load from the actual T0 dataset
    # For demonstration, we'll simulate with realistic examples
    
    t0_templates = [
        "Given the context: {context}, answer the question: {question}",
        "Is the following statement true or false? {statement}",
        "Choose the correct answer: {question} A) {option_a} B) {option_b} C) {option_c}",
        "Complete the analogy: {analogy}",
        "Rewrite the following sentence to be more {style}: {sentence}",
        "What is the main idea of the following passage? {passage}",
        "Predict what happens next in this scenario: {scenario}",
        "Explain why {event} happened",
        "Generate a {creative_type} on the topic of {topic}",
        "Fill in the blank: {sentence}"
    ]
    
    contexts = [
        "The Industrial Revolution was a period of major industrialization and innovation during the late 1700s and early 1800s.",
        "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar.",
        "Machine learning is a branch of artificial intelligence that focuses on building applications that learn from data and improve their accuracy over time."
    ]
    
    questions = [
        "What were the major effects of this period?",
        "How does this process benefit plants?",
        "What distinguishes supervised from unsupervised learning?"
    ]
    
    statements = [
        "Water boils at 100 degrees Celsius at sea level.",
        "The Great Wall of China is visible from space with the naked eye.",
        "Humans use only 10% of their brains.",
        "Albert Einstein failed mathematics in school."
    ]
    
    multiple_choice = [
        {"q": "Which planet is closest to the sun?", "a": "Venus", "b": "Mercury", "c": "Earth"},
        {"q": "Who wrote 'Romeo and Juliet'?", "a": "Charles Dickens", "b": "William Shakespeare", "c": "Jane Austen"},
        {"q": "What is the capital of Australia?", "a": "Sydney", "b": "Melbourne", "c": "Canberra"}
    ]
    
    analogies = [
        "Bird is to sky as fish is to ___",
        "Doctor is to hospital as teacher is to ___",
        "Pen is to write as knife is to ___"
    ]
    
    styles = ["formal", "concise", "persuasive", "descriptive", "technical", "simple"]
    
    sentences = [
        "The cat sat on the mat.",
        "It was a dark and stormy night.",
        "The team worked hard to finish the project."
    ]
    
    passages = [
        "The internet has transformed how we communicate, access information, and conduct business. It has connected people across the globe, democratized knowledge, and created new economic opportunities.",
        "Climate change is altering ecosystems worldwide. Rising temperatures are causing polar ice to melt, leading to rising sea levels. Extreme weather events are becoming more common and more severe."
    ]
    
    scenarios = [
        "A student discovers a new mathematical theorem that contradicts established principles.",
        "A company announces it has developed a battery that lasts 10 times longer than current technology.",
        "Scientists detect an unusual radio signal from a distant star system."
    ]
    
    events = [
        "the dinosaurs became extinct",
        "the Great Depression occurred",
        "the Internet was developed",
        "certain species evolved to have bright coloration"
    ]
    
    creative_types = ["poem", "short story", "essay", "dialogue", "speech"]
    
    topics = ["freedom", "technology", "nature", "equality", "progress", "harmony", "conflict"]
    
    fill_blanks = [
        "In economics, supply and demand are factors that determine the ___ of goods and services.",
        "The three branches of the US government are executive, legislative, and ___.",
        "The process by which plants make their own food using sunlight is called ___."
    ]
    
    instructions = []
    for _ in range(SAMPLE_SIZE):
        template = random.choice(t0_templates)
        
        if "{context}" in template and "{question}" in template:
            idx = random.randint(0, min(len(contexts), len(questions)) - 1)
            instruction = template.format(context=contexts[idx], question=questions[idx])
        elif "{statement}" in template:
            instruction = template.format(statement=random.choice(statements))
        elif "{option_a}" in template:
            mc = random.choice(multiple_choice)
            instruction = template.format(question=mc["q"], option_a=mc["a"], option_b=mc["b"], option_c=mc["c"])
        elif "{analogy}" in template:
            instruction = template.format(analogy=random.choice(analogies))
        elif "{style}" in template:
            instruction = template.format(style=random.choice(styles), sentence=random.choice(sentences))
        elif "{passage}" in template:
            instruction = template.format(passage=random.choice(passages))
        elif "{scenario}" in template:
            instruction = template.format(scenario=random.choice(scenarios))
        elif "{event}" in template:
            instruction = template.format(event=random.choice(events))
        elif "{creative_type}" in template:
            instruction = template.format(creative_type=random.choice(creative_types), topic=random.choice(topics))
        elif "Fill in the blank" in template:
            instruction = template.format(sentence=random.choice(fill_blanks))
            
        instructions.append(instruction)
    
    return instructions
