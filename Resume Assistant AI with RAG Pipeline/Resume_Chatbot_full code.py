
import pdfplumber
from pydantic import BaseModel, Field, ValidationError
import re

class ResumeChunk(BaseModel):
    section: str = Field(..., description="Section of the resume (e.g., Experience, Education)")
    text: str = Field(..., min_length=10, description="Chunk of resume text")
    chunk_id: str

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text.strip()

def chunk_resume_with_labels(resume_text):
    section_headers = ["SUMMARY", "EXPERIENCE", "EDUCATION", "SKILLS & TOOLS", "CERTIFICATIONS", "PROJECTS"]
    section_pattern = re.compile(rf"(?i)^\s*({'|'.join(section_headers)})\s*$", re.MULTILINE)
    print(section_pattern)
    print('------------------------')
    sections = section_pattern.split(resume_text)
    chunks = []
    current_section = "Unknown"
    for i in range(len(sections)):
        text_content = sections[i].strip()
        if text_content in section_headers:
            current_section = text_content
        elif text_content:
            try:
                chunk = ResumeChunk(section=current_section, text=text_content, chunk_id=f"chunk-{i//2}")
                chunks.append(chunk)
            except ValidationError as e:
                print(f"Skipping invalid chunk: {e}")
    return chunks

pdf_path = "D:\\GenAI\\code\\data\\Gowtham Mani Resume.pdf"
resume_text = extract_text_from_pdf(pdf_path)
chunks = chunk_resume_with_labels(resume_text)

chunks

new_chunks = []
for i in chunks:
    new_chunks.append(i.text)
new_chunks

from langchain.embeddings import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = embedding_model.embed_documents(new_chunks)

Final_emb = []
for i in range(len(new_chunks)):
    d = {}
    d['id'] = str(i)
    d['values'] = embeddings[i]
    l = {}
    l[chunks[i].section] = chunks[i].text
    d['metadata'] = l
    Final_emb.append(d)

for i in Final_emb:
    print(i)

from pinecone import Pinecone
pcobj = Pinecone(api_key="pcsk_4zeATK_9n9GkNeC2xEJAzgL45EKQ6FNjVSngrrBSE24XETgqHPYu1kxpHDDTTVGwbp68c4")
index = pcobj.Index('firstrag')

from langchain_pinecone import PineconeVectorStore
vector_store = PineconeVectorStore(index=index, embedding=embedding_model, namespace="Gowtham3rd Resume")

index.upsert(Final_emb)

GROQ_API_KEY = "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"

from langchain_groq import ChatGroq
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama3-8b-8192",
    temperature=0.1,
    max_retries=2,
)

query = 'tell about the education'
query_emb = embedding_model.embed_query(query)

result1 = index.query(vector=query_emb, top_k=3, include_metadata=True)
result1

rt = []
for i in range(len(result1['matches'])):
    d = result1['matches'][i]['metadata']
    for key, value in d.items():
        rt.append(value)
rt

context = "\n".join(["".join(text) for text in rt])
context

prompt = f"""
context:
{context}

Question:
{query}
"""

llm.invoke([{"role": "ai", "content": prompt}]).content

while True:
    user_query = input("Your query: ")
    if user_query.lower() == "exit":
        print("Exiting chatbot.")
        break
    query_emb = embedding_model.embed_query(user_query)
    result1 = index.query(vector=query_emb, top_k=3, include_metadata=True)
    rt = []
    for i in range(len(result1['matches'])):
        d = result1['matches'][i]['metadata']
        for key, value in d.items():
            rt.append(value)
    context = "\n".join(["".join(text) for text in rt])
    prompt = f"""
    Content:
    {context}

    Question:
    {user_query}
    """
    print(llm.invoke([{"role": "user", "content": prompt}]).content)