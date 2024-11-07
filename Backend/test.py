import os
import csv

# Paths to your dataset files
dataset_folder = "/Users/nathannguyen/Documents/RAG_BOT_1/Backend/Data/msmarco"
collection_file = os.path.join(dataset_folder, "collection.tsv")
qrels_file = os.path.join(dataset_folder, "qrels.train.tsv")

# Load a small set of document IDs from collection.tsv
print("Loading passage IDs from collection.tsv...")
collection_ids = set()
with open(collection_file, 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    for i, line in enumerate(reader):
        if len(line) < 2:
            continue
        passage_id = line[0]
        collection_ids.add(passage_id)
        if i >= 10000:  # Limit to the first 10,000 passages for testing
            break
print(f"Loaded {len(collection_ids)} passage IDs from collection.tsv")

# Check if qrels document IDs exist in collection_ids
print("Checking if qrels document IDs exist in collection...")
missing_ids = set()
with open(qrels_file, 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    for line in reader:
        if len(line) < 4:
            continue
        query_id, _, passage_id, _ = line
        if passage_id not in collection_ids:
            missing_ids.add(passage_id)

# Report results
if missing_ids:
    print(f"Found {len(missing_ids)} qrels document IDs missing in collection.tsv")
    print("Sample missing IDs:", list(missing_ids)[:10])  # Display first 10 missing IDs
else:
    print("All qrels document IDs found in collection.tsv")