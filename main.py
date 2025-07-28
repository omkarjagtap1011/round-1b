import os
import json
from llama_cpp import Llama
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from sentence_transformers.cross_encoder import CrossEncoder
from datetime import datetime
from extract_struct import extractor
import shutil

# --- Configuration ---
INPUT_DIR = "input"
OUTPUT_DIR = "output"
JSON_DIR = "json"
MODEL_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
MODEL_FILE = "tinyllama-1.1b-chat-v1.0.Q5_K_S.gguf"
MODEL_CACHE = "./models"

def parse_input_json(input_path: str):
    with open(os.path.join(INPUT_DIR, input_path), "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = [doc["filename"] for doc in data["documents"]]
    persona = data["persona"]["role"]
    jtbd = data["job_to_be_done"]["task"]

    return {
        "documents": documents,
        "persona": persona,
        "job_to_be_done": jtbd,
        "prompt": f"As {persona}, {jtbd}"
    }

def generate_output_json(documents, persona, job_to_be_done, ranked_sections, subsection_analyses, output_path="challenge1b_output.json"):
    output = {
        "metadata": {
            "input_documents": documents,
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": datetime.utcnow().isoformat()
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }

    for rank, section in enumerate(ranked_sections, start=1):
        output["extracted_sections"].append({
            "document": section["metadata"]["source"].replace(".json", ".pdf"),
            "section_title": section["metadata"]["section"],
            "importance_rank": rank,
            "page_number": section["metadata"]["page"]
        })

    for analysis in subsection_analyses:
        output["subsection_analysis"].append({
            "document": analysis["metadata"]["source"],
            "refined_text": analysis["refined_text"],
            "page_number": analysis["metadata"]["page"]
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"✅ Output saved to {output_path}")


def re_rank_cross_encoders(documents, prompt):
    ranked_sections = []
    subsection_analyses = []

    docs = [d["text"] for d in documents]
    encoder_model = CrossEncoder("./models/encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder_model.rank(prompt, docs, top_k=5)

    for rank in ranks:
        doc = documents[rank["corpus_id"]]
        ranked_sections.append({
            "text": doc["text"],
            "metadata": doc["metadata"]
        })
        subsection_analyses.append({
            "refined_text": doc["text"],
            "metadata": doc["metadata"]
        })

    return ranked_sections, subsection_analyses


def get_vector_collection() -> chromadb.Collection:
    """Gets or creates a ChromaDB collection for vector storage.

    Creates an Ollama embedding function using the nomic-embed-text model and initializes
    a persistent ChromaDB client. Returns a collection that can be used to store and
    query document embeddings.

    Returns:
        chromadb.Collection: A ChromaDB collection configured with the Ollama embedding
            function and cosine similarity space.
    """
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )

    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )

# --- Load and Parse JSON Chunks ---
def load_chunks_from_json_dir(json_dir):
    chunks = []
    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            path = os.path.join(json_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for idx, section in enumerate(data):
                    text = section.get("content", "").strip()
                    section_title = section.get("section-header", "")
                    page = section.get("page", -1)
                    if text:
                        chunks.append({
                            "id": f"{filename}_{idx}",
                            "document": text,
                            "metadata": {
                                "section": section_title,
                                "page": page,
                                "source": filename
                            }
                        })
    return chunks

# --- Add Chunks to Vector DB ---
def index_chunks(collection, chunks, batch_size=50):
    print(f"📥 Indexing {len(chunks)} chunks into ChromaDB in batches of {batch_size}...")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        documents = [c["document"] for c in batch]
        metadatas = [c["metadata"] for c in batch]
        ids = [c["id"] for c in batch]

        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        print(f"✅ Indexed batch {i // batch_size + 1} ({i + len(batch)}/{len(chunks)})")

    print("✅ All chunks indexed.")


# --- Query Vector DB ---
def query_top_chunks(collection, prompt, top_k=10):
    result = collection.query(query_texts=[prompt], n_results=top_k)

    retrieved = []
    for doc, meta in zip(result["documents"][0], result["metadatas"][0]):
        retrieved.append({
            "text": doc,
            "metadata": meta
        })
    return retrieved


# --- Main ---
if __name__ == "__main__":
    if os.path.exists("demo-rag-chroma"):
        shutil.rmtree("demo-rag-chroma")

    # Persona setup
    parsed_input = parse_input_json("challenge1b_input.json")

    extractor(parsed_input["documents"])

    print("Loading structured chunks...")
    chunks = load_chunks_from_json_dir(JSON_DIR)
    print(f"Loaded {len(chunks)} chunks")

    collection = get_vector_collection()

    if not collection.count():
        index_chunks(collection, chunks) 
    else:
        print("ℹ️ Collection already has data. Skipping indexing.")

    print("🔍 Performing semantic search...")
    top_chunks = query_top_chunks(collection, parsed_input["prompt"], top_k=10)
    
    # encoder_model = CrossEncoder("./models")
    # sections, text = re_rank_cross_encoders(top_chunks, parsed_input["prompt"])

    # with open("challenge1b_output.json", "w", encoding="utf-8") as f:
    #     f.write(str(relevant_text))

    sections, analyses = re_rank_cross_encoders(top_chunks, parsed_input["prompt"])

    generate_output_json(
        documents=parsed_input["documents"],
        persona=parsed_input["persona"],
        job_to_be_done=parsed_input["job_to_be_done"],
        ranked_sections=sections,
        subsection_analyses=analyses
    )


    print(f"\n✅ Output saved to challenge1b_output.json")

    shutil.rmtree("json")