import re
from typing import List

def get_chunks(text: str, chunk_size: int = 750, chunk_overlap: int = 150) -> List[str]: #
    """
    Splits text into logical chunks, prioritizing paragraphs and then sentences.
    It attempts to maintain context by using overlap and merging small chunks.

    Args:
        text (str): The input document text.
        chunk_size (int): Desired maximum chunk size (approximate, in characters).
        chunk_overlap (int): Overlap between chunks (in characters).

    Returns:
        List[str]: A list of text chunks.
    """
    if not text:
        return []

    # First attempt: split by double newlines (paragraphs)
    paragraphs = re.split(r'\n\n+', text)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If adding the current paragraph keeps the chunk within size, add it
        if len(current_chunk) + len(para) + 2 < chunk_size: # +2 for newline if current_chunk exists
            current_chunk += (("\n\n" if current_chunk else "") + para)
        else:
            # If current_chunk exists, finalize it
            if current_chunk:
                chunks.append(current_chunk)
            
            # Start a new chunk with the current paragraph
            current_chunk = para

            # If the paragraph itself is too large, split it further by sentences
            while len(current_chunk) > chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', current_chunk)
                sub_chunk = ""
                processed_len = 0
                for i, sent in enumerate(sentences):
                    sent = sent.strip()
                    if not sent:
                        continue

                    if len(sub_chunk) + len(sent) + 1 < chunk_size: # +1 for space
                        sub_chunk += (( " " if sub_chunk else "") + sent)
                        processed_len += len(sent) + (1 if sub_chunk else 0)
                    else:
                        if sub_chunk:
                            chunks.append(sub_chunk)
                            # Add overlap for the next sub_chunk from the end of the current sub_chunk
                            overlap_text = sub_chunk[-chunk_overlap:] if len(sub_chunk) >= chunk_overlap else sub_chunk
                            sub_chunk = overlap_text + (( " " if overlap_text else "") + sent)
                            processed_len = len(sent) + (len(overlap_text) + 1 if overlap_text else 0)
                        else: # If a single sentence is larger than chunk_size, take a slice
                            chunks.append(sent[:chunk_size])
                            sub_chunk = sent[chunk_size - chunk_overlap:] # overlap from the chopped part
                            processed_len = len(sub_chunk)
                
                # After processing all sentences in the current_chunk (paragraph),
                # if there's a remaining sub_chunk, add it and reset current_chunk.
                if sub_chunk:
                    if sub_chunk not in chunks: # Avoid duplicate if it was just added
                        chunks.append(sub_chunk)
                current_chunk = "" # Reset current_chunk as it has been fully processed by sentences

    if current_chunk: # Add any remaining current_chunk
        chunks.append(current_chunk)

    # Post-processing: Merge very small chunks if they are adjacent and can fit
    final_chunks = []
    if chunks:
        final_chunks.append(chunks[0])
        for i in range(1, len(chunks)):
            # Try to merge with previous chunk if it doesn't exceed chunk_size too much
            # Allow slightly larger for merging to avoid too many tiny chunks
            if len(final_chunks[-1]) + len(chunks[i]) + 2 <= chunk_size * 1.2: # Note: 1.2 allows chunks to exceed chunk_size slightly for merging
                final_chunks[-1] += ("\n\n" + chunks[i])
            else:
                final_chunks.append(chunks[i])

    return [chunk.strip() for chunk in final_chunks if chunk.strip()] # Filter empty chunks and strip

# Example Usage (for testing)
if __name__ == "__main__":
    long_text = """
    This is the first paragraph. It contains several sentences.
    This paragraph talks about a very important policy detail.

    This is the second paragraph. It also has multiple sentences.
    For example, this sentence specifically discusses insurance coverage for pre-existing conditions.
    Another sentence here details the grace period for premium payments.

    A third, very long paragraph follows, which might exceed the chunk size. This paragraph contains
    extensive information about legal compliance in the insurance domain, including various
    regulatory frameworks and specific clauses. It goes on and on, detailing procedures for
    claims, exclusions, and dispute resolution mechanisms. This single paragraph is intentionally
    made to be quite lengthy to test the sentence-level splitting fallback. It ensures that even
    if a paragraph is massive, it will be broken down into more manageable, semantically
    meaningful units while maintaining context through overlap.

    Here is a small table:
    Header1 | Header2
    --------|--------
    Data1   | Data2
    Data3   | Data4

    End of document.
    """
    chunks = get_chunks(long_text, chunk_size=1000, chunk_overlap=200) # Ensure this matches default for testing
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} (Length: {len(chunk)}) ---")
        print(chunk)
        print("\n")