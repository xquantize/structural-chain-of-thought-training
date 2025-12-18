# Structural Chain-of-Thought Training for Chart Understanding

## Overview

This repository implements a training methodology for vision-language models that explicitly decomposes chart structure before answering questions. We hypothesize that training models to reason through intermediate structural representations improves performance on chart question answering tasks.

## Research Question

Can training vision-language models to explicitly generate structural descriptions and reasoning steps before answering improve performance on chart understanding tasks compared to standard end-to-end training?

## Approach

### Problem
Current vision-language models are trained end-to-end on chart question answering, treating the reasoning process as a black box. Charts contain explicit structure (axes, legends, data regions) that could be leveraged during reasoning.

### Method
We modify the training procedure to include intermediate reasoning steps:

**Standard training:**
```
Input: [chart image] + question
Output: answer
```

**Approach:**
```
Input: [chart image] + question
Output: <structure>chart description</structure>
         <reasoning>step-by-step reasoning</reasoning>
         <answer>final answer</answer>
```

### Implementation
1. Generate structural annotations using a base VLM
2. Fine-tune the model using LoRA on the structured format
3. Evaluate on ChartQA benchmark
4. Compare against baseline direct-answer training

## Dataset

**ChartQA** (Ahmed Masry et al.)
- 28,299 training examples
- 2,500 test examples
- Charts with questions requiring visual reasoning
