# What are Evals (Evaluations)?

An AI evaluation (or "eval" for short) is a systematic test or metric used to measure an AI model's performance, accuracy, quality, safety, and reliability (kind of like unit tests for LLM outputs). Evals determine if the AI meets quality standards by comparing outputs against expected results to catch hallucinations, bias, or performance drops. They are crucial for moving from experiemental prototypes to reliable, deployed AI products.

An eval typically involves a prompt, an output form the model, and scoring logic that grades that response.

## How about Evals in Agentic AI and RAG?

In agentic AI, evals measure how effectively an autonomous system (agent) reasons plans, and uses tools to complete multi-turn tasks. Instead of just checking final output, agentic evals track the following:

- Tool Correctness: Does the agent select and use the right tools? (i.e. search, api calls)
- Reasoning Capability: Is the reasoning chain logical?
- Goal Completion: Did the agent accomplish the overall objective?
- Multi-turn Coherence: Can it maintain context over long, interactive loops?

In RAG, evals assess how well an application retrieves external knowledge and generates answers based on that context. It focuses on two main pillars:

- Retrieval Evaluation: Did the system find the right information?
- Generation Evaluation (Faithfulness/Groundedness): Is the answer supported by the retrieved context, or is it a hallucination?

## What are some key types of Evals?

- LLM-as-a-Judge: Using a stronger AI to evaluate a smaller one.
- Offline Eval: Testing on static datasets to compare against expected results.
- Behavioral/Adversarial Eval: Testing for edge cases or, in agentic contexts, trying to make the agent break.
