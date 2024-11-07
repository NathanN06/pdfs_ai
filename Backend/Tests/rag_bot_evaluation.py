from tqdm.auto import tqdm
import pandas as pd
import json
import os
import random
from openai import OpenAI
from typing import Optional
from dotenv import load_dotenv
from datasets import load_dataset, Dataset
import re
from services.document_loader import load_documents
from services.retrieval_service import retrieve_documents
from services.indexing_service import load_index

# Load environment variables from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pd.set_option("display.max_colwidth", None)

# Load documents
DATA_FOLDER = os.getenv("DATA_FOLDER")
documents = load_documents(DATA_FOLDER)  # Load documents from your data folder

pd.set_option("display.max_colwidth", None)

# Load dataset from Hugging Face
ds = load_dataset("m-ric/huggingface_doc", split="train")

# Import custom chunking functions
from utils.chunking import (
    chunk_text,
    semantic_chunk_text,
    recursive_chunk_text,
    adaptive_chunk_text,
    chunk_by_paragraph,
    chunk_by_tokens,
)
from langchain.docstore.document import Document as LangchainDocument

# Map chunking methods to function names
chunking_methods = {
    "character": chunk_text,
    "semantic": semantic_chunk_text,
    "recursive": recursive_chunk_text,
    "adaptive": adaptive_chunk_text,
    "paragraph": chunk_by_paragraph,
    "tokens": chunk_by_tokens,
}

# Choose a chunking method
chosen_method = "character"
chunk_function = chunking_methods[chosen_method]

# Define chunking parameters
chunk_size = 2000
chunk_overlap = 200

# Convert documents and apply chunking
langchain_docs = [LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in tqdm(ds)]
docs_processed = []

for doc in langchain_docs:
    try:
        chunks = chunk_function(doc.page_content, chunk_size=chunk_size, overlap=chunk_overlap)
    except TypeError:
        chunks = chunk_function(doc.page_content)

    docs_processed += [LangchainDocument(page_content=chunk, metadata=doc.metadata) for chunk in chunks]

# Save the processed documents to the specified directory
output_path = "/Users/nathannguyen/Documents/RAG_BOT_1/Backend/processed_docs.json"
docs_json = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs_processed]

with open(output_path, "w") as f:
    json.dump(docs_json, f)
print(f"Processed documents saved to {output_path}")

# Refined QA Generation Prompt
QA_generation_prompt = """
Your task is to write a specific, fact-based question and an answer based on the context provided. 
Make sure the question is clear and answerable without relying on vague or unclear references.
Use complete, standalone phrasing in the question itself.

Provide your answer as follows:

Output:::
Factoid question: (your factoid question)
Answer: (your answer to the factoid question)

Here is the context.

Context: {context}
Output:::
"""

# Function to call OpenAI GPT API
def call_llm(prompt: str):
    response = client.chat.completions.create(model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=500)
    return response.choices[0].message.content

# Generate QA pairs with the new prompt
N_GENERATIONS = 10  # Number of QA pairs to generate
outputs = []
for sampled_context in tqdm(random.sample(docs_processed, N_GENERATIONS)):
    output_QA_couple = call_llm(QA_generation_prompt.format(context=sampled_context.page_content))

    # Parsing logic for QA pairs
    try:
        question_start = output_QA_couple.find("Factoid question:")
        answer_start = output_QA_couple.find("Answer:")

        if question_start == -1 or answer_start == -1:
            print("Parsing error.")
            continue

        question = output_QA_couple[question_start + len("Factoid question:"):answer_start].strip()
        answer = output_QA_couple[answer_start + len("Answer:"):].strip()

        if question and answer and len(answer) < 300:
            outputs.append({
                "context": sampled_context.page_content,
                "question": question,
                "answer": answer,
                "source_doc": sampled_context.metadata["source"],
            })
    except Exception as e:
        print(f"Error processing output: {e}")
        continue

# Critique Prompts
question_groundedness_critique_prompt = """
You will be given a context and a question.
Provide a 'total rating' on how well the question can be answered using only the provided context. 
Rate on a scale of 1 to 5, where 1 means "not answerable at all," and 5 means "clearly answerable."

Answer as follows:

Answer:::
Evaluation: (your rationale for the rating)
Total rating: (1-5)

Here are the question and context.

Question: {question}\n
Context: {context}\n
Answer::: 
"""

question_relevance_critique_prompt = """
You will be given a question.
Provide a 'total rating' of how useful this question would be for developers in the NLP field, especially those working with Hugging Face.
Rate from 1 to 5, where 1 is "not useful at all," and 5 is "extremely useful."

Answer as follows:

Answer:::
Evaluation: (your rationale for the rating)
Total rating: (1-5)

Question: {question}\n
Answer::: 
"""

question_standalone_critique_prompt = """
You will be given a question.
Rate how well the question stands alone without additional context, from 1 to 5. 
A 5 means it is entirely understandable independently, while a 1 means it heavily relies on additional context.

Answer as follows:

Answer:::
Evaluation: (your rationale for the rating)
Total rating: (1-5)

Question: {question}\n
Answer::: 
"""

# Evaluation for generated QA pairs
print("Generating critique for each QA couple...")
for output in tqdm(outputs):
    evaluations = {
        "groundedness": call_llm(
            question_groundedness_critique_prompt.format(context=output["context"], question=output["question"]),
        ),
        "relevance": call_llm(
            question_relevance_critique_prompt.format(question=output["question"]),
        ),
        "standalone": call_llm(
            question_standalone_critique_prompt.format(question=output["question"]),
        ),
    }
    try:
        for criterion, evaluation in evaluations.items():
            score_match = re.search(r"Total rating:\s*(\d+)", evaluation)
            eval_match = re.search(r"Evaluation:\s*(.*)", evaluation)

            if score_match and eval_match:
                score = int(score_match.group(1))
                eval_text = eval_match.group(1).strip()

                output.update({
                    f"{criterion}_score": score,
                    f"{criterion}_eval": eval_text,
                })
    except Exception as e:
        print(f"Error processing evaluation: {e}")
        continue

# Display final evaluations and filter
generated_questions = pd.DataFrame(outputs)

print("Evaluation dataset before filtering:")
print(generated_questions[["question", "answer", "groundedness_score", "relevance_score", "standalone_score"]])

# Filter questions with high scores in all three criteria
# Remove filtering temporarily to examine all outputs
generated_questions = generated_questions.loc[
    (generated_questions["groundedness_score"] >= 2)
    & (generated_questions["relevance_score"] >= 2)
    & (generated_questions["standalone_score"] >= 2)
]

print("Final evaluation dataset:")
print(generated_questions[["question", "answer", "groundedness_score", "relevance_score", "standalone_score"]])

# Save evaluation dataset to Hugging Face's Dataset format
eval_dataset = Dataset.from_pandas(generated_questions, preserve_index=False)

from services.retrieval_service import retrieve_documents
from services.indexing_service import load_index

# Define RAG testing function
from services.embedding_service import embed_query

# Define RAG testing function
def run_rag_tests(
    eval_dataset,
    llm,
    output_file: str,
    verbose: Optional[bool] = True,
    test_settings: Optional[str] = None,
):
    """Runs RAG tests on the given dataset using backend retrieval and indexing functions."""
    # Load or initialize output storage
    try:
        with open(output_file, "r") as f:
            outputs = json.load(f)
    except FileNotFoundError:
        outputs = []

    # Load the FAISS index using your backend function
    knowledge_index = load_index()  # This loads the index as per your backend setup

    def generate_answer(llm, question, relevant_docs):
        context = "\n\n".join([doc[1] for doc in relevant_docs])  # Concatenate document text for context
        prompt = f"Given the following context:\n{context}\n\nAnswer the question: {question}"
        response = llm(messages=[{"role": "user", "content": prompt}])
        return response["choices"][0]["message"]["content"]

    for example in tqdm(eval_dataset):
        question = example["question"]
        
        # Skip questions already processed
        if question in [output["question"] for output in outputs]:
            continue

        # Retrieve relevant documents using your custom retrieval method
        relevant_docs = retrieve_documents(knowledge_index, question, documents)

        # Generate an answer using the language model
        answer = generate_answer(llm, question, relevant_docs)

        if verbose:
            print("=======================================================")
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f'True answer: {example["answer"]}')
        
        # Save the result
        result = {
            "question": question,
            "true_answer": example["answer"],
            "source_doc": example["source_doc"],
            "generated_answer": answer,
            "retrieved_docs": [doc for doc in relevant_docs],
        }
        if test_settings:
            result["test_settings"] = test_settings
        outputs.append(result)

        # Save output results to file
        with open(output_file, "w") as f:
            json.dump(outputs, f)


# EVALUATION_PROMPT definition
EVALUATION_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
[Is the response correct, accurate, and factual based on the reference answer?]
Score 1: The response is completely incorrect, inaccurate, and/or not factual.
Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
Score 3: The response is somewhat correct, accurate, and/or factual.
Score 4: The response is mostly correct, accurate, and factual.
Score 5: The response is completely correct, accurate, and factual.

###Feedback:"""

# Set up prompt template for evaluation
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain_openai import ChatOpenAI

evaluation_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a fair evaluator language model."),
        HumanMessagePromptTemplate.from_template(EVALUATION_PROMPT),
    ]
)

# Initialize the evaluation model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
eval_chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
evaluator_name = "GPT4"

# Define function to evaluate answers
def evaluate_answers(
    answer_path: str,
    eval_chat_model,
    evaluator_name: str,
    evaluation_prompt_template: ChatPromptTemplate,
) -> None:
    """Evaluates generated answers. Modifies the given answer file in place for better checkpointing."""
    answers = []
    if os.path.isfile(answer_path):  # load previous generations if they exist
        answers = json.load(open(answer_path, "r"))

    for experiment in tqdm(answers):
        if f"eval_score_{evaluator_name}" in experiment:
            continue

        eval_prompt = evaluation_prompt_template.format_messages(
            instruction=experiment["question"],
            response=experiment["generated_answer"],
            reference_answer=experiment["true_answer"],
        )
        eval_result = eval_chat_model.invoke(eval_prompt)
        feedback, score = [item.strip() for item in eval_result.content.split("[RESULT]")]
        experiment[f"eval_score_{evaluator_name}"] = score
        experiment[f"eval_feedback_{evaluator_name}"] = feedback

        with open(answer_path, "w") as f:
            json.dump(answers, f)

# Example loop to run tests across configurations
if not os.path.exists("./output"):
    os.mkdir("./output")

for chunk_size in [200]:  # Modify based on your chunk size preference
    for embeddings in ["thenlper/gte-small"]:  # Modify based on your embedding model
        for rerank in [True, False]:  # Choose if reranking should be enabled
            settings_name = f"chunk:{chunk_size}_embeddings:{embeddings.replace('/', '~')}_rerank:{rerank}"
            output_file_name = f"./output/rag_{settings_name}.json"

            print(f"Running evaluation for {settings_name}:")

            print("Loading knowledge base embeddings...")
            knowledge_index = load_index()  # Load index from your backend

            print("Running RAG...")
            reranker = None  # Modify based on reranking model, if applicable

            run_rag_tests(
                eval_dataset=eval_dataset,
                llm=eval_chat_model,
                output_file=output_file_name,
                verbose=False,
                test_settings=settings_name,    
            )

            print("Running evaluation...")
            evaluate_answers(
                output_file_name,
                eval_chat_model,
                evaluator_name,
                evaluation_prompt_template,
            )

import glob
import plotly.express as px

# Aggregate results and visualize
outputs = []
for file in glob.glob("./output/*.json"):
    output = pd.DataFrame(json.load(open(file, "r")))
    output["settings"] = file
    outputs.append(output)

# Check if outputs is empty
if not outputs:
    print("No data to concatenate in `outputs` list.")
    exit()
else:
    result = pd.concat(outputs)

fig = px.bar(
    result,
    x="settings",
    y="eval_score_GPT4",
    color="eval_score_GPT4",
    labels={"value": "Accuracy", "settings": "Configuration"},
    color_continuous_scale="bluered",
)
fig.update_layout(
    width=1000,
    height=600,
    barmode="group",
    yaxis_range=[0, 100],
    title="<b>Accuracy of different RAG configurations</b>",
    xaxis_title="RAG settings",
    font=dict(size=15),
)
fig.layout.yaxis.ticksuffix = "%"
fig.update_coloraxes(showscale=False)
fig.update_traces(texttemplate="%{y:.1f}", textposition="outside")
fig.show()