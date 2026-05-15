import re

from src.core.retrieval import extract_keywords


def generate_extractive_answer(question, relevant_chunks, max_sentences=3):
    """Build a short grounded answer by selecting the best matching sentences."""
    keywords = set(extract_keywords(question))
    sentence_candidates = []

    for chunk in relevant_chunks:
        sentences = re.split(r"(?<=[.!?])\s+", chunk["text"])
        for sentence in sentences:
            if len(sentence.strip()) < 20:
                continue
            words = set(re.findall(r"[a-zA-Z0-9]+", sentence.lower()))
            overlap = len(keywords.intersection(words)) if keywords else 0
            sentence_candidates.append((overlap, sentence.strip(), chunk["source"]))

    sentence_candidates.sort(key=lambda item: item[0], reverse=True)
    selected = sentence_candidates[:max_sentences]

    if not selected:
        return "I could not find an exact sentence-level answer in the indexed documents.", []

    answer = " ".join([item[1] for item in selected])
    sources = [item[2] for item in selected]
    return answer, sources