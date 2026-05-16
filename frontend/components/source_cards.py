#source iteam means the source of the information used to generate the answer, such as a document or a webpage. The source item typically includes a preview of the content from that source, which can help users understand where the information is coming from and assess its relevance and credibility.
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