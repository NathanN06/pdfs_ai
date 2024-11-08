import nltk
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Load sentence embedding model

def calculate_sentence_similarity(sent1, sent2):
    emb1 = embedding_model.encode(sent1)
    emb2 = embedding_model.encode(sent2)
    return cosine_similarity([emb1], [emb2])[0][0]

def dynamic_overlap(sentences, default_overlap, threshold=0.7):
    overlap = []
    for i in range(1, default_overlap + 1):
        if i < len(sentences) and calculate_sentence_similarity(sentences[-1], sentences[-i]) > threshold:
            overlap.append(sentences[-i])
    return overlap

def chunk_text(text, max_length=650, default_overlap=3):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_word_count = sum(len(w) + 1 for w in current_chunk)
        if current_word_count + len(word) <= max_length:
            current_chunk.append(word)
        else:
            overlap = current_chunk[-default_overlap:]
            chunks.append(" ".join(current_chunk))
            current_chunk = overlap + [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def semantic_chunk_text(text, max_length=600, default_overlap=3):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for i, sentence in enumerate(sentences):
        if current_length + len(sentence) <= max_length:
            current_chunk.append(sentence)
            current_length += len(sentence) + 1
        else:
            overlap = dynamic_overlap(current_chunk, default_overlap)
            chunks.append(" ".join(current_chunk))
            current_chunk = overlap + [sentence]
            current_length = sum(len(s) for s in current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def recursive_chunk_text(text, max_length=600, min_length=250, depth=0, max_depth=4, default_overlap=3):
    if len(text) <= max_length or depth >= max_depth:
        return [text]

    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        if current_length + len(sentence) <= max_length:
            current_chunk.append(sentence)
            current_length += len(sentence) + 1
        else:
            if current_length >= min_length:
                overlap = dynamic_overlap(current_chunk, default_overlap)
                chunks.append(" ".join(current_chunk).strip())
                current_chunk = overlap + [sentence]
                current_length = sum(len(s) for s in current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_length:
            final_chunks.extend(recursive_chunk_text(chunk, max_length, min_length, depth + 1, max_depth, default_overlap))
        else:
            final_chunks.append(chunk)

    return final_chunks

def chunk_by_paragraph(text, max_length=700, default_overlap=1):
    paragraphs = re.split(r'\n\s*\n|\n{2,}|\r\n\r\n', text.strip())
    paragraphs = [para.strip() for para in paragraphs if para.strip()]
    chunks = []
    current_chunk = []

    for paragraph in paragraphs:
        if len(paragraph) > max_length:
            sentences = nltk.sent_tokenize(paragraph)
            sentence_chunk = []
            for sentence in sentences:
                if sum(len(s) for s in sentence_chunk) + len(sentence) <= max_length:
                    sentence_chunk.append(sentence)
                else:
                    overlap = dynamic_overlap(sentence_chunk, default_overlap)
                    chunks.append(" ".join(sentence_chunk).strip())
                    sentence_chunk = overlap + [sentence]
            if sentence_chunk:
                chunks.append(" ".join(sentence_chunk).strip())
        else:
            if sum(len(p) for p in current_chunk) + len(paragraph) > max_length:
                overlap = dynamic_overlap(current_chunk, default_overlap)
                chunks.append(" ".join(current_chunk).strip())
                current_chunk = overlap
            current_chunk.append(paragraph)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return chunks

def adaptive_chunk_text(text, default_max_length=700, default_overlap=3):
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

        if current_length + sentence_length <= max_length:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            overlap = dynamic_overlap(current_chunk, default_overlap)
            chunks.append(" ".join(current_chunk))
            current_chunk = overlap + [sentence]
            current_length = sum(len(s.split()) for s in current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def chunk_by_tokens(text, max_tokens=650, buffer=15, default_overlap=25):
    tokens = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for token in tokens:
        if current_length < max_tokens:
            current_chunk.append(token)
            current_length += 1
        else:
            overlap = current_chunk[-default_overlap:]
            chunks.append(" ".join(current_chunk))
            current_chunk = overlap + [token]
            current_length = len(current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks