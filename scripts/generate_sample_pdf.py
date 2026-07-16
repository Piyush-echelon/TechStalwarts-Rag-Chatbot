"""
generate_sample_pdf.py — creates sample.pdf for the demo conversation.

Run: python scripts/generate_sample_pdf.py
Output: docs/sample.pdf
"""

from pathlib import Path

try:
    from fpdf import FPDF
except ImportError:
    raise SystemExit("Install fpdf2 first: pip install fpdf2")

PAGES = [
    (
        "Introduction to Artificial Intelligence",
        """Artificial Intelligence (AI) refers to the simulation of human intelligence
in machines that are programmed to think and learn. The term was coined by John
McCarthy in 1956 at the Dartmouth Conference, which is considered the birthplace
of AI as a field.

AI systems can perform tasks that typically require human intelligence, including
visual perception, speech recognition, decision-making, and language translation.
Modern AI is powered primarily by machine learning and deep learning techniques.

Key branches of AI include:
- Machine Learning (ML): systems that learn from data
- Natural Language Processing (NLP): understanding and generating text
- Computer Vision (CV): interpreting images and video
- Robotics: physical AI agents that interact with the world""",
    ),
    (
        "Machine Learning Fundamentals",
        """Machine Learning is a subset of AI that enables systems to learn from
experience without being explicitly programmed. It was formally defined by
Arthur Samuel in 1959.

There are three main types of machine learning:

1. Supervised Learning: The model is trained on labelled data. Examples include
   linear regression (for predicting house prices) and decision trees (for
   classification tasks).

2. Unsupervised Learning: The model finds hidden patterns in unlabelled data.
   K-means clustering and PCA are common examples.

3. Reinforcement Learning: An agent learns by interacting with an environment
   and receiving rewards or penalties. Used in game-playing AI like AlphaGo.

Key metrics for evaluating ML models include accuracy, precision, recall, F1
score, and AUC-ROC. Overfitting and underfitting are the two primary failure
modes to guard against.""",
    ),
    (
        "Large Language Models and RAG",
        """Large Language Models (LLMs) are neural networks trained on massive corpora
of text data using the transformer architecture, first described in the 2017
paper "Attention Is All You Need" by Vaswani et al.

Modern LLMs such as GPT-4, Claude, and Gemini contain hundreds of billions of
parameters and demonstrate remarkable capabilities in reasoning, code generation,
and language understanding.

Retrieval-Augmented Generation (RAG) is a technique that addresses the knowledge
limitations of LLMs by retrieving relevant external documents at query time and
injecting them into the model's context. This allows the LLM to answer questions
about private or up-to-date information without retraining.

Key components of a RAG system:
- PDF/document ingestion and text extraction
- Text chunking and embedding generation
- Vector store for semantic search
- LLM prompt construction and generation
- Source attribution and citation

RAG significantly reduces hallucination compared to pure LLM inference because
the model can ground its answers in retrieved evidence.""",
    ),
    (
        "Ethical Considerations in AI",
        """As AI systems become increasingly capable and pervasive, ethical considerations
have moved to the forefront of the field. Key concerns include:

Bias and Fairness: AI models trained on biased data can perpetuate and amplify
social inequalities. For example, facial recognition systems have been shown to
perform significantly worse on darker-skinned individuals.

Transparency and Explainability: Many modern AI systems, especially deep learning
models, are "black boxes" - their internal reasoning is opaque. Explainable AI
(XAI) is an active research area focused on making model decisions interpretable.

Privacy: Large models may memorise sensitive training data. Techniques such as
differential privacy and federated learning aim to train models without exposing
individual data.

AI Safety: As AI systems become more autonomous, ensuring they behave as intended
and do not cause unintended harm is critical. Alignment research focuses on
encoding human values into AI systems.

Regulation: Governments worldwide are developing frameworks to govern AI use.
The EU AI Act (2024) is the most comprehensive regulatory framework to date.""",
    ),
]


def main():
    out_path = Path(__file__).parent.parent / "docs" / "sample.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    for title, body in PAGES:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 12, title, ln=True)
        pdf.ln(4)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 7, body)

    pdf.output(str(out_path))
    print(f"Sample PDF written to: {out_path}")
    print(f"Pages: {len(PAGES)}")


if __name__ == "__main__":
    main()
