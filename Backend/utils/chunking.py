import nltk
import re

def chunk_text(text, max_length=512):
    """Original chunking method that splits text into chunks based on word count."""
    words = text.split()  # Split the text into a list of words
    chunks = []  # Initialize an empty list to hold the resulting chunks of text
    current_chunk = []  # Initialize an empty list to accumulate words for the current chunk

    for word in words:
        current_chunk.append(word)  # Add the word to the current chunk
        if len(" ".join(current_chunk)) >= max_length:  # If the combined length reaches the limit
            chunks.append(" ".join(current_chunk))  # Join the words and add to chunks
            current_chunk = []  # Reset current_chunk for the next chunk

    # Add any leftover words as the final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks  # Return the list of text chunks

def semantic_chunk_text(text, max_length):
    """Chunk text semantically using sentence tokenization."""
    sentences = nltk.sent_tokenize(text)  # Use 'punkt' tokenizer
    chunks = []
    current_chunk = []

    for sentence in sentences:
        # Check if adding the current sentence exceeds the max length
        if len(" ".join(current_chunk + [sentence])) <= max_length:
            current_chunk.append(sentence)  # Add to the current chunk
        else:
            # Append the current chunk and start a new one
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]  # Start a new chunk with the current sentence

    # Add the last chunk if there are any sentences left
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def recursive_chunk_text(text, max_length):
    """Recursively chunk text into smaller segments based on max_length."""
    if len(text) <= max_length:
        return [text]

    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_length:  # +1 for the space
            current_chunk += (sentence + " ")  # Add the sentence to the current chunk
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def chunk_by_paragraph(text, max_length=512):
    """Attempts to chunk by paragraph, defaults to sentences if no paragraph markers are found."""
    # Split by paragraphs first
    paragraphs = re.split(r'\n\s*\n|\n{2,}|\r\n\r\n', text.strip())
    paragraphs = [para.strip() for para in paragraphs if para.strip()]
    
    # If only one large chunk was created, default to sentence-based chunking
    if len(paragraphs) == 1:
        print("No paragraphs detected; switching to sentence-based chunking.")
        sentences = nltk.sent_tokenize(text)
        chunks, current_chunk = [], []

        # Accumulate sentences until reaching max_length
        for sentence in sentences:
            if len(" ".join(current_chunk) + sentence) > max_length:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
            current_chunk.append(sentence)
        
        # Append any remaining sentences
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    else:
        return paragraphs

def adaptive_chunk_text(text, default_max_length=512):
    """Adaptive chunking based on the content's complexity and structure."""
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        
        # Determine max length based on sentence complexity
        if sentence_length > 50:
            max_length = default_max_length // 2  # Smaller chunk size for complex sentences
        else:
            max_length = default_max_length  # Default chunk size for simpler sentences

        # If the current chunk is empty, start a new chunk with the current sentence
        if not current_chunk:
            current_chunk.append(sentence)
            current_length = sentence_length
            continue
        
        # Check if adding the current sentence exceeds the max length
        if current_length + sentence_length + 1 <= max_length:  # +1 for the space
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            # Check if current chunk is too small to be useful
            if current_length < 20:  # Arbitrary threshold for minimum chunk size
                # If the current chunk is too small, add the next sentence anyway
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                # Append the current chunk and start a new one
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]  # Start a new chunk with the current sentence
                current_length = sentence_length  # Reset the current length to the new chunk

    # Add the last chunk if there are any sentences left
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def chunk_by_tokens(text, max_tokens=512):
    """Splits the input text into chunks based on a fixed number of tokens."""
    tokens = text.split()
    chunks = []
    current_chunk = []

    for token in tokens:
        current_chunk.append(token)
        if len(current_chunk) >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks