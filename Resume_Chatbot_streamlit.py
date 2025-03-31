
import streamlit as st
import pdfplumber
from pydantic import BaseModel, Field, ValidationError
import re
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq

class ResumeChunk(BaseModel):
    section: str = Field(..., description="Section of the resume (e.g., Experience, Education)")
    text: str = Field(..., min_length=10, description="Chunk of resume text")
    chunk_id: str

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text.strip()

def chunk_resume_with_labels(resume_text):
    section_headers = ["SUMMARY", "EXPERIENCE", "EDUCATION", "SKILLS & TOOLS", "CERTIFICATIONS", "PROJECTS"]
    section_pattern = re.compile(rf"(?i)^\s*({'|'.join(section_headers)})\s*$", re.MULTILINE)
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
                st.write(f"Skipping invalid chunk: {e}")
    return chunks

st.title("Resume Chatbot")

pdf_file = st.file_uploader("Upload Resume PDF", type=["pdf"])
if pdf_file is not None:
    resume_text = extract_text_from_pdf(pdf_file)
    chunks = chunk_resume_with_labels(resume_text)
    new_chunks = [chunk.text for chunk in chunks]
    
    st.write("Processing resume and creating embeddings...")
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
    
    st.write("Connecting to Pinecone and upserting embeddings...")
    pcobj = Pinecone(api_key="*****")
    index = pcobj.Index('firstrag')
    index.upsert(Final_emb)
    
    st.write("Initializing LLM...")
    GROQ_API_KEY = "*******"
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama3-8b-8192",
        temperature=0.1,
        max_retries=2,
    )
    
    user_query = st.text_input("Enter your query about the resume")
    if st.button("Submit Query") and user_query:
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
        answer = llm.invoke([{"role": "user", "content": prompt}]).content
        st.write("Answer:")
        st.write(answer)
else:
    st.write("Please upload a resume PDF to start.")
