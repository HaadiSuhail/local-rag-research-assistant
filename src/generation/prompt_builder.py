from typing import List, Dict


class PromptBuilder:
    def __init__(self):
        pass

    def build(
        self,
        query: str,
        retrieved_chunks: List[Dict]
    ) -> str:
        """
        Constructs a RAG prompt for Mistral.
        """

        context_blocks = []

        for i, chunk in enumerate(retrieved_chunks):
            context_blocks.append(
                f"[Document {i+1} | Score: {chunk['score']:.3f}]\n{chunk['text']}\n"
            )

        context_text = "\n\n".join(context_blocks)

        prompt = f"""<s>[INST]
You are a research assistant.

Use ONLY the provided context to answer the question.
Cite document numbers in brackets like [1], [2] when using information.
If the answer is not in the context, say you don't know.

Before answering:
1. Identify the key concepts relevant to the question.
2. Explain the mechanism or reasoning described in the context.
3. Synthesize the ideas into a structured answer.

Provide a well-structured explanation with clear logical flow.

Context:
{context_text}

Question:
{query}
[/INST]"""

        return prompt