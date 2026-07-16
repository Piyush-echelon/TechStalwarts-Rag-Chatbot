# Demo Conversation — Advanced RAG Chatbot

**Document used:** `docs/sample.pdf`  
**Pages:** 4 (Introduction to AI, Machine Learning, LLMs & RAG, Ethics)

---

## Query 1 — Factual in-scope question

**User:** What is Artificial Intelligence and who coined the term?

**Assistant:**
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines
that are programmed to think and learn. The term was coined by **John McCarthy in 1956**
at the Dartmouth Conference, which is considered the birthplace of AI as a field.

AI systems can perform tasks that typically require human intelligence, including visual
perception, speech recognition, decision-making, and language translation.

[Sources: Page 1]

---

## Query 2 — Multi-concept question

**User:** What are the three main types of machine learning?

**Assistant:**
According to the document, there are three main types of machine learning:

1. **Supervised Learning** — The model is trained on labelled data. Examples include
   linear regression (for predicting house prices) and decision trees (for
   classification tasks).

2. **Unsupervised Learning** — The model finds hidden patterns in unlabelled data.
   K-means clustering and PCA are common examples.

3. **Reinforcement Learning** — An agent learns by interacting with an environment
   and receiving rewards or penalties. Used in game-playing AI like AlphaGo.

[Sources: Page 2]

---

## Query 3 — Technical deep-dive

**User:** How does RAG reduce hallucination in LLMs?

**Assistant:**
RAG (Retrieval-Augmented Generation) significantly reduces hallucination compared to
pure LLM inference because the model can **ground its answers in retrieved evidence**.

Instead of relying solely on the model's parametric memory (which may be outdated or
incorrect), RAG retrieves relevant external documents at query time and injects them
into the model's context. This allows the LLM to answer based on actual source material
rather than generating text from internal weights alone.

[Sources: Page 3]

---

## Query 4 — Ethical topic (still in-scope)

**User:** What is the EU AI Act and when was it introduced?

**Assistant:**
The **EU AI Act (2024)** is described in the document as the most comprehensive
regulatory framework for AI governance to date. It was developed as part of a broader
global effort by governments to create frameworks that govern AI use.

The document situates the EU AI Act within the larger context of AI ethics, specifically
in the section on **Regulation** under ethical considerations for AI systems.

[Sources: Page 4]

---

## Query 5 — Out-of-scope question (graceful refusal)

**User:** What is the stock price of NVIDIA today?

**Assistant:**
I don't have enough information in the provided document to answer this question.
Please try rephrasing or ask about a topic covered in the document.

The uploaded document covers topics such as AI fundamentals, machine learning,
large language models, RAG architecture, and AI ethics — but does not contain
financial or market data.

[Sources: none]

---

*This demo was generated using `docs/sample.pdf` with OpenAI GPT-4o-mini as the LLM
and ChromaDB with hybrid (dense + BM25) retrieval.*
