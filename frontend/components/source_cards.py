# Build small source preview cards for the answer panel.
def prepare_source_items(relevant_chunks):
    """Create deduplicated source preview items for the answer panel."""
    source_items = []
    seen = set()

    for chunk in relevant_chunks:
        src = chunk["source"]
        if src in seen:
            continue

        seen.add(src)
        preview = " ".join(chunk["text"].split())[:240]
        if len(preview) == 240:
            preview += "..."

        source_items.append({"source": src, "preview": preview})

    return source_items