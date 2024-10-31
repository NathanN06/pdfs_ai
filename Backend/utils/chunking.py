import nltk
import re
from sklearn.metrics.pairwise import cosine_similarity

def chunk_text(text, max_length=512):
    """Character-based chunking method that splits text into chunks based on a maximum character count without splitting words."""
    words = text.split()  # Split the text into individual words
    chunks = []  # Initialize list to hold text chunks
    current_chunk = ""  # Initialize current chunk as an empty string

    for word in words:
        # Check if adding the next word would exceed max_length
        if len(current_chunk) + len(word) + 1 <= max_length:  # +1 for the space
            current_chunk += (" " + word) if current_chunk else word  # Add word with a space if chunk is non-empty
        else:
            # If chunk is filled, add it to the list and reset current chunk
            chunks.append(current_chunk)
            current_chunk = word  # Start new chunk with the current word

    # Add any remaining content as the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def semantic_chunk_text(text, max_length=512):
    """Chunk text semantically using sentence tokenization while preserving sentence boundaries and semantic coherence."""
    sentences = nltk.sent_tokenize(text)  # Tokenize by sentence
    chunks = []  # List to hold semantic chunks
    current_chunk = []  # Temporary list for sentences in the current chunk
    current_length = 0  # Track the character length of the current chunk

    for sentence in sentences:
        sentence_length = len(sentence)
        
        # Check if adding the current sentence exceeds the max length
        if current_length + sentence_length <= max_length or not current_chunk:
            current_chunk.append(sentence)  # Add sentence to the chunk
            current_length += sentence_length + 1  # Update length (+1 for space)
        else:
            # If adding exceeds max, finalize current chunk and start a new one
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]  # New chunk starts with current sentence
            current_length = sentence_length  # Reset length for new chunk

    # Add any remaining sentences as the final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def recursive_chunk_text(text, max_length=512, min_length=100, depth=0, max_depth=5):
    
    """Recursively chunk text into segments based on max_length, with adaptive splitting to preserve coherence and prevent excessive recursion."""
    
    # Base case: if text length is within max_length or max recursion depth is reached
    if len(text) <= max_length or depth >= max_depth:
        return [text]

    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        # Accumulate sentences until reaching max_length
        if sum(len(s) for s in current_chunk) + len(sentence) + len(current_chunk) <= max_length:
            current_chunk.append(sentence)
        else:
            # Avoid chunks that are too short by adding an extra sentence if needed
            if len(" ".join(current_chunk)) < min_length and sentence:
                current_chunk.append(sentence)
            # Append the current chunk and start a new one
            chunks.append(" ".join(current_chunk).strip())
            current_chunk = [sentence]  # Start a new chunk

    # Append any remaining sentences in the final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    # Recursively split any chunk that exceeds max_length
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_length:
            # Increase recursion depth
            final_chunks.extend(recursive_chunk_text(chunk, max_length, min_length, depth + 1, max_depth))
        else:
            final_chunks.append(chunk)

    return final_chunks

def chunk_by_paragraph(text, max_length=512):
    """Attempts to chunk by paragraph, with additional splitting for long paragraphs and grouping smaller ones."""
    # Split by paragraph markers
    paragraphs = re.split(r'\n\s*\n|\n{2,}|\r\n\r\n', text.strip())
    paragraphs = [para.strip() for para in paragraphs if para.strip()]
    chunks = []
    current_chunk = []

    for paragraph in paragraphs:
        if len(paragraph) > max_length:
            # Further split the paragraph by sentences if it exceeds max_length
            sentences = nltk.sent_tokenize(paragraph)
            sentence_chunk = []
            for sentence in sentences:
                if sum(len(s) for s in sentence_chunk) + len(sentence) + len(sentence_chunk) <= max_length:
                    sentence_chunk.append(sentence)
                else:
                    chunks.append(" ".join(sentence_chunk).strip())
                    sentence_chunk = [sentence]
            # Add any remaining sentences in the sentence_chunk
            if sentence_chunk:
                chunks.append(" ".join(sentence_chunk).strip())
        else:
            # If the current chunk plus the paragraph exceeds max_length, finalize the current chunk
            if sum(len(p) for p in current_chunk) + len(paragraph) + len(current_chunk) > max_length:
                chunks.append(" ".join(current_chunk).strip())
                current_chunk = []
            current_chunk.append(paragraph)
    
    # Add any remaining paragraphs in the current_chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return chunks

def adaptive_chunk_text(text, default_max_length=512):
    """Adaptive chunking based on sentence complexity and structure."""
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        
        # Dynamically adjust max length based on sentence complexity
        if sentence_length > 50:
            max_length = default_max_length // 2  # Smaller chunks for complex sentences
        elif sentence_length > 30:
            max_length = int(default_max_length * 0.75)  # Mid-size chunks for moderately complex sentences
        else:
            max_length = default_max_length  # Default chunk size for simpler sentences

        # Start a new chunk if current one is empty
        if not current_chunk:
            current_chunk.append(sentence)
            current_length = sentence_length
            continue
        
        # Check if adding the sentence exceeds max length
        if current_length + sentence_length + 1 <= max_length:  # +1 for space
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            # If the chunk size is very small, merge with the next sentence
            if current_length < default_max_length // 4:  # Threshold for minimum chunk size
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                # Finalize the current chunk and start a new one
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length

    # Append the last chunk if there are sentences left
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def chunk_by_tokens(text, max_tokens=512, buffer=20):
    """
    Splits the input text into chunks based on a fixed number of tokens with a buffer.
    Allows some flexibility around max_tokens to avoid cutting off mid-sentence.
    """
    tokens = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for token in tokens:
        current_chunk.append(token)
        current_length += 1

        # Check if current chunk is within max_tokens but allows buffer overflow
        if current_length >= max_tokens:
            # If within buffer, allow a few extra tokens to complete the sentence
            if current_length - max_tokens <= buffer:
                continue  # Allow buffer overflow, continue adding tokens
            
            # If buffer exceeded, end the chunk here
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

    # Append any remaining tokens as the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks