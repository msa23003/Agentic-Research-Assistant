# Agentic AI: RAG System with Langflow

## Overview
The *Agentic AI RAG System* is a locally hosted Retrieval-Augmented Generation (RAG) system powered by fine-tuned open-source language models (LLMs). This project utilized research papers from arXiv's CVPR category to assist users in learning, reasoning, and performing advanced tasks. With seamless integration into Langflow, the system delivered intuitive workflows and agentic behavior, making it a powerful tool for research and exploration.



## Objective
Built an *Agentic AI-based Retrieval-Augmented Generation (RAG)* system that:
- Fine-tuned open-source language models for domain-specific tasks.
- Enabled fast, accurate retrieval of CVPR paper data.
- Simplified research through summaries, explanations, and comparisons.
- Used Langflow to enhance interactivity with agentic capabilities.



## Key Components

### 1. *LLM Development*
- Fine-tuned open-source LLMs (e.g., Llama, Qwen, Mistral) using lightweight methods like *LoRA* or *QLoRA*.
- Conducted experiments on *Kaggle Notebook*.
- Served the model locally using frameworks like *Ollama*.
- Optimized for performance and resource efficiency.

### 2. *Langflow Integration*
- Designed workflows in *Langflow* to interact with the Agentic AI RAG system.
- Built a RAG pipeline using arXiv CVPR research papers.
- Integrated vector storage for fast and efficient document retrieval.
- Developed tools to:
  - Summarize papers.
  - Compare methodologies and results.
  - Explain complex concepts.

### 3. *RAG System Development*
- Extracted metadata (e.g., title, authors, abstract) from CVPR papers.
- Generated *document embeddings* for retrieval.
- Used a vector database (e.g., Qdrant, Pinecone) for accurate searches.
- Optimized retrieval for relevance and speed.

### 4. *Agentic AI Capabilities*
- Provided clear explanations of:
  - Research methodologies.
  - Results and contributions.
- Summarized and compared multiple papers.
- Simplified complex concepts for better understanding.
- Executed multi-step tasks with robust reasoning and chaining tools.

### 5. *Evaluation and Testing*
- Defined metrics to measure system performance and response quality.
- Tested for relevance, coherence, and accuracy.



## Deliverables
- *Fine-tuned LLM*: Domain-specific language model optimized for local deployment.
- *Langflow Workflows*: Interactive workflows integrated with the Agentic AI RAG pipeline.
- *RAG System*: Advanced document interaction capabilities.
- *Evaluation Report*: Detailed metrics on system performance.
- *Deployment-Ready System*: Tested with CVPR papers for a local playground environment.



## Installation
### Prerequisites
- *Python 3.8+*
- *Pip* or *Conda* for dependencies
- *Langflow* installed locally
- Vector database (e.g., Qdrant, Pinecone, Milvus)
- GPU for fine-tuning (optional)

### Steps
1. Clone the repository:
    bash
    git clone https://github.com/msa23003/agentic-ai-rag-system.git
    cd agentic-ai-rag-system
    
2. Install dependencies:
    bash
    pip install -r requirements.txt
    
3. Set up vector storage (e.g., Qdrant or Pinecone).
4. Preprocessed the arXiv CVPR dataset using provided scripts.
5. Fine-tuned the LLM using *LoRA* or *QLoRA*.
6. Launched the system:
    bash
    python app.py
    



## Usage
- *Interactive Workflows*: Used Langflow for intuitive interaction.
- *Document Search*: Retrieved and explored CVPR papers using the RAG pipeline.
- *Agentic AI Tools*:
  - Summarized, compared, and explained research papers.
  - Simplified complex methodologies.
  - Planned and executed multi-step reasoning tasks.



## Evaluation Metrics
- *Performance*: Response time and resource efficiency.
- *Relevance*: Accuracy and coherence of results.
- *Quality*: Clarity and correctness of responses.



## Contributions
Contributions to improve this project were welcomed. Please follow the [CONTRIBUTING.md](CONTRIBUTING.md) guidelines for pull requests.



## License
This project was licensed under the [MIT License](LICENSE).



## Acknowledgments
- arXiv for access to CVPR research papers.
- Open-source contributors for LLM frameworks.
- Langflow for enabling intuitive workflows.
