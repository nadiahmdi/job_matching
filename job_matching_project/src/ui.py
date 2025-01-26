# Import Libraries
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
from src.embeddings import generate_embedding
from src.matching import calculate_similarity
from src.preprocess import load_data

# Load the data
resumes, job_descriptions = load_data("data/resumes.csv", "data/job_descriptions.csv")

st.title("Job Matching System")

# Input: Job description
job_description = st.text_area("Enter Job Description")

if st.button("Find Matches"):
    # Generate job description embedding
    job_embedding = generate_embedding(job_description)

    # Generate resume embeddings
    resume_embeddings = resumes['Resume Text'].apply(generate_embedding).tolist()

    # Calculate similarity scores
    scores = calculate_similarity(job_embedding, resume_embeddings)

    # Add scores to resumes dataframe
    resumes['Alignment Score'] = scores

    # Sort resumes by score
    top_resumes = resumes.sort_values(by='Alignment Score', ascending=False).head(5)

    # Display top matches
    st.subheader("Top Matching Resumes")
    for idx, row in top_resumes.iterrows():
        st.write(f"**Resume {idx+1}:**")
        st.write(row['Resume Text'])
        st.write(f"**Score:** {row['Alignment Score']:.2f}")
