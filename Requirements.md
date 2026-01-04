## Problem Statement 2:

Modern AI assistants use Retrieval-Augmented Generation (RAG), where large language models generate responses based on documents retrieved from sources such as PDFs, webpages, and internal enterprise data.

However, even with retrieval in place, generated responses may contain fabricated information, rely on incomplete or loosely relevant documents, or combine unrelated content from multiple sources. In many cases, correct information exists in the knowledge base but is not effectively surfaced or utilized during response generation. These limitations highlight the need to improve retrieval logic so that LLMs are provided with coherent, complete, and contextually relevant evidence before generating responses, particularly in enterprise and regulated environments where reliability and explainability are critical.

For this problem, candidates should consider a corpus consisting of the top 10 research
papers on Large Language Models (LLMs).

### Instructions:
1. Design and implement an improved retrieval strategy for a Retrieval-Augmented Generation (RAG) system.
2. Compare the proposed retrieval strategy against a baseline retrieval approach using the same queries and knowledge base.
3. A brief description of the proposed retrieval approach
4. A comparison of retrieved documents for the same query before and after applying the improved retrieval
5. An evaluation summary showing how the improved retrieval impacts the quality of retrieved context and the reliability of generated responses
6. The final submission should be presented in a structured format (for example: table or short report) that clearly demonstrates the improvement over baseline.

Additional Points
-  Use a simple embedding-based retrieval approach (e.g., cosine similarity over dense embeddings) as the baseline. There is no requirement to compare against a specific external RAG model or pipeline.
- Improvement should be demonstrated by comparing baseline and improved retrieval using the same queries and corpus. Evaluation may include fact-level validation against authoritative public sources (such as Wikipedia or the original research papers), along with qualitative or LLM-assisted analysis. No specific metric is mandatory, but the evaluation should clearly show improved relevance, grounding, and reliability.