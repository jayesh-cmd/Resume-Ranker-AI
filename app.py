import streamlit as st
import PyPDF2
import io
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Resume Ranker AI")

job_description = st.text_area("Enter Job Description")
uploaded_files = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

if st.button("Rank Resumes"):
    resumes = []
    filenames = []
    resume_bytes_list = []

    for uploaded_file in uploaded_files:
        pdf_bytes = uploaded_file.read()
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        text = ''.join(page.extract_text() for page in reader.pages)
        resumes.append(text)
        filenames.append(uploaded_file.name)
        resume_bytes_list.append(pdf_bytes)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    job_emb = model.encode(job_description)
    resume_embs = [model.encode(r) for r in resumes]
    scores = [cosine_similarity([job_emb], [emb])[0][0] for emb in resume_embs]

    ranked = sorted(zip(filenames, scores, resume_bytes_list), key=lambda x: x[1], reverse=True)

    st.write("### Top Ranked Candidates:")
    for i, (filename, score, file_bytes) in enumerate(ranked, start=1):
        st.write(f"**{i}. {filename} → Similarity: {score:.2f}**")
        st.download_button("View PDF",
                           data=file_bytes,
                           file_name=filename,
                           mime='application/pdf')
