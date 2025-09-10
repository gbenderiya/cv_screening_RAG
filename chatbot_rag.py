from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from uuid import uuid4
from openai import OpenAI
from langchain_core.documents import Document
import gradio as gr
import os, re, requests

# load .env for HF_KEY
from dotenv import load_dotenv
load_dotenv()

# --- config ---
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get("HF_KEY"),
)

# --- Chroma collections ---
cv_store = Chroma(
    collection_name="cv_collection",
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)

job_store = Chroma(
    collection_name="job_collection",
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)

# --- helpers ---
def extract_job_id(url: str):
    match = re.search(r'/job/([^/?#]+)', url)
    return match.group(1) if match else None

def fetch_job_from_url(url):
    job_id = extract_job_id(url)
    api_url = f"https://new-api.zangia.mn/api/jobs/{job_id}"
    resp = requests.get(api_url)
    resp.raise_for_status()
    d = resp.json()
    return {
        "title": d.get("title", ""),
        "description": d.get("description", ""),
        "requirements": d.get("requirements", ""),
        "skills": [s.strip().lower() for s in d.get("skills", [])],
        "additional": d.get("additional", "")
    }

# --- ingestion ---
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=100)

def ingest_cvs():
    all_chunks = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(DATA_PATH, filename)
            print(f"Loading CV: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            raw_docs = loader.load()
            chunks = splitter.split_documents(raw_docs)
            for c in chunks:
                c.metadata["filename"] = filename
            all_chunks.extend(chunks)
    if all_chunks:
        uuids = [str(uuid4()) for _ in range(len(all_chunks))]
        cv_store.add_documents(all_chunks, ids=uuids)
        print("CVs added")

def ingest_job(url):
    job_data = fetch_job_from_url(url)
    text = f"""
    Title: {job_data['title']}
    Description: {job_data['description']}
    Requirements: {job_data['requirements']}
    Skills: {', '.join(job_data['skills'])}
    Additional: {job_data['additional']}
    """
    
    doc = Document(
        page_content=text,
        metadata={"url": url, "title": job_data['title']}
    )
    uuids = [str(uuid4())]
    job_store.add_documents(documents=[doc], ids=uuids)
    print(f"✅ Job {job_data['title']} added")


# --- evaluation ---
def evaluate_cv_against_job(cv_doc, job_doc):
    prompt = f"""
    Compare the following CV and Job description. 
    Give scores and a short reasoning.

    Job Title: {job_doc.metadata.get("title")}
    Job Posting: {job_doc.page_content}
    Candidate Name: {cv_doc.metadata.get("filename")}
    Candidate CV: {cv_doc.page_content}

    Return in JSON format:
    {{
        "skills_score": 0-10,
        "experience_score": 0-10,
        "overall_fit": 0-10,
        "explanation": "..."
    }}
    """
    resp = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0,
    )
    return resp.choices[0].message.content

def match_job_to_cvs(job_url, k=3):
    # try retrieving job from DB
    job_docs = job_store.similarity_search(job_url, k=1)
    if not job_docs:
        ingest_job(job_url)
        job_docs = job_store.similarity_search(job_url, k=1)

    job_doc = job_docs[0]
    ingest_cvs()
    all_cv_docs = cv_store.get()["documents"]

    results = []
    for doc, meta in zip(all_cv_docs, cv_store.get()["metadatas"]):
        cv_doc = Document(page_content=doc, metadata=meta)
        evaluation = evaluate_cv_against_job(cv_doc, job_doc)
        results.append(f"📄 {cv_doc.metadata.get('filename')} → {evaluation}")
    return "\n\n".join(results)

# --- chatbot ---
def respond(message, history):
    if "zangia.mn/job" in message:   # detect job link
        answer = match_job_to_cvs(message, k=3)
        yield history + [{"role": "user", "content": message},
                         {"role": "assistant", "content": answer}]
    else:
        yield history + [{"role": "user", "content": message},
                         {"role": "assistant", "content": "Please provide a Zangia job link to evaluate candidates."}]

# --- Gradio UI ---
with gr.Blocks() as app:
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(placeholder="Paste job link or ask a question...")

    msg.submit(respond, inputs=[msg, chatbot], outputs=chatbot)

app.launch(share=True)
