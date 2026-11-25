import re


def chunk_pages(
    pages,
    document_id,
    chunk_words=250,
    overlap_words=40,
):
    if not document_id:
        raise ValueError("document_id must be provided")
    if chunk_words < 1:
        raise ValueError("chunk_words must be positive")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be less than chunk_words")

    chunks = []
    for page in pages:
        page_number = int(page["page"])
        words = re.findall(r"\S+", page["text"])
        step = chunk_words - overlap_words
        for start in range(0, len(words), step):
            selected = words[start:start + chunk_words]
            if not selected:
                continue
            chunk_number = start // step + 1
            chunks.append(
                {
                    "id": "{}:p{}:c{}".format(
                        document_id,
                        page_number,
                        chunk_number,
                    ),
                    "chunk_id": "{}:p{}:c{}".format(
                        document_id,
                        page_number,
                        chunk_number,
                    ),
                    "document_id": document_id,
                    "page": page_number,
                    "text": " ".join(selected),
                }
            )
            if start + chunk_words >= len(words):
                break
    return chunks
