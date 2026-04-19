# What is Retrieval-Augmented Generation (RAG)?

Retrieval-Augmented Generation (RAG) is an AI framework that improves Large Language Model (LLM) accuracy by retrieving external, up-to-date data before generating a response.

## Why is RAG useful?

- **Reduced Hallucinations**: Answers are grounded in provided facts, not just probabilistic guesses.
- **Up-to-date Information**: Access to real-time data without retraining the AI model.
- **Source Attribution**: RAG allows the AI to provide citations to the original documents.
- **Data Privacy**: Allows firms to use propriety data securely without feeding it into public LLM training.

## How does RAG work?

RAG connects LLMs to external, proprietary, or current datasets (like vector databases) instead of relying solely on pre-trained knowledge. It works by:

- Retrieval: Finding relevant documents related to a user query.
- Augmentation: Feeding those documents + query to the LLM.
- Generation: Producing a grounded, accurate response.

## What is naive RAG vs production RAG?

Naive RAG is a basic, quick-start, single-pass retrieval, while Production RAG uses complex, multi-stage, and adaptive techniques for higher accuracy.

In naive RAG, we would do and have the following:

- **Retrieval Steps**: single-step vector search
- **Handling Data**: simple embedding/linear search
- **Best For**: prototypes, simple Q&A

In production RAG, we would do and have the following:

- **Retrieval Steps**: multi-stage (i.e. query decommposition and/or transformation, hybrid search)
- **Handling Data**: advanced chunking and re-ranking
- **Best For**: real-world apps (i.e. legal, healthcare)

## More about production RAG

- **Query Transformation**: Production RAG transforms user queries (rewriting, decomposition) before searching.
- **Re-ranking**: Production systems use models to rank the most relevant information after retrieval to eliminate noise.
- **Hybrid Search**: Combining semantic (semantic) and keyword search to improve retrieval relevance.

## Additional Notes

For complex applications, an "Agentic" RAG approach allows the model to decide when and how to search, further increasing accuracy.
