import nltk
import re
from sklearn.metrics.pairwise import cosine_similarity

def chunk_text(text, max_length=512, overlap=50):
    """Character-based chunking method that splits text into chunks based on a maximum character count without splitting words."""
    words = text.split()
    chunks = []
    current_chunk = ""

    for word in words:
        if len(current_chunk) + len(word) + 1 <= max_length:
            current_chunk += (" " + word) if current_chunk else word
        else:
            chunks.append(current_chunk)
            current_chunk = " ".join(current_chunk.split()[-overlap:]) + " " + word

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def semantic_chunk_text(text, max_length=512, overlap=50):
    """Chunk text semantically using sentence tokenization while preserving sentence boundaries and semantic coherence."""
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length <= max_length or not current_chunk:
            current_chunk.append(sentence)
            current_length += sentence_length + 1
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:] + [sentence]
            current_length = sum(len(s) for s in current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def recursive_chunk_text(text, max_length=512, min_length=100, depth=0, max_depth=5, overlap=50):
    """Recursively chunk text into segments with adaptive splitting to maintain coherence and minimize recursion."""
    # Base case: if text length is within limits or max recursion depth reached
    if len(text) <= max_length or depth >= max_depth:
        return [text]

    # Tokenize sentences only once
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0  # Track length to avoid repeated sum calls

    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length <= max_length:
            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 accounts for space between sentences
        else:
            # Ensure the chunk meets the minimum length
            if current_length < min_length:
                current_chunk.append(sentence)
                current_length += sentence_length + 1
            # Finalize current chunk
            chunks.append(" ".join(current_chunk).strip())
            current_chunk = current_chunk[-overlap:] + [sentence]  # Retain overlap
            current_length = sum(len(s) for s in current_chunk)  # Recalculate for overlap

    # Append any remaining sentences in the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    # Recursively process chunks that exceed max_length
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_length:
            final_chunks.extend(recursive_chunk_text(chunk, max_length, min_length, depth + 1, max_depth, overlap))
        else:
            final_chunks.append(chunk)

    return final_chunks

def chunk_by_paragraph(text, max_length=512, overlap=50):
    """Attempts to chunk by paragraph, with additional splitting for long paragraphs and grouping smaller ones."""
    paragraphs = re.split(r'\n\s*\n|\n{2,}|\r\n\r\n', text.strip())
    paragraphs = [para.strip() for para in paragraphs if para.strip()]
    chunks = []
    current_chunk = []

    for paragraph in paragraphs:
        if len(paragraph) > max_length:
            sentences = nltk.sent_tokenize(paragraph)
            sentence_chunk = []
            for sentence in sentences:
                if sum(len(s) for s in sentence_chunk) + len(sentence) + len(sentence_chunk) <= max_length:
                    sentence_chunk.append(sentence)
                else:
                    chunks.append(" ".join(sentence_chunk).strip())
                    sentence_chunk = sentence_chunk[-overlap:] + [sentence]
            if sentence_chunk:
                chunks.append(" ".join(sentence_chunk).strip())
        else:
            if sum(len(p) for p in current_chunk) + len(paragraph) + len(current_chunk) > max_length:
                chunks.append(" ".join(current_chunk).strip())
                current_chunk = current_chunk[-overlap:]
            current_chunk.append(paragraph)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return chunks

def adaptive_chunk_text(text, default_max_length=512, overlap=50):
    """Adaptive chunking based on sentence complexity and structure."""
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if sentence_length > 50:
            max_length = default_max_length // 2
        elif sentence_length > 30:
            max_length = int(default_max_length * 0.75)
        else:
            max_length = default_max_length

        if not current_chunk:
            current_chunk.append(sentence)
            current_length = sentence_length
            continue

        if current_length + sentence_length + 1 <= max_length:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            if current_length < default_max_length // 4:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = current_chunk[-overlap:] + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def chunk_by_tokens(text, max_tokens=512, buffer=20, overlap=50):
    """Splits the input text into chunks based on a fixed number of tokens with a buffer, allowing some flexibility to avoid mid-sentence cuts."""
    tokens = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for token in tokens:
        current_chunk.append(token)
        current_length += 1

        if current_length >= max_tokens:
            if current_length - max_tokens <= buffer:
                continue
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:]
            current_length = len(current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks