import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import time
import logging

# Configure logging (place this after imports)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import streamlit as st
import random
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px 
import re
from Preprocessing.document_processor import CVProcessor
from Preprocessing.vector_store import CVVectorStore
from RAG.rag_engine import EnhancedRAGEngine
from RAG.job_matcher import EnhancedJobMatcher
from RAG.cv_summarizer import CVSummarizer
from RAG.job_recommender import JobRecommender
from RAG.hr_question_generator import HRQuestionGenerator


def app():   

    # --- Theme Colors ---
    theme = {
        "primary": "#017691",       # Blue
        "primary2": "#7DBBC9",       # Blue
        "primary3": "#A1CFDD",       # Blue

        "secondary": "#FF9F1C",     # Orange
        "accent": "#e0e0e0",
        "background": "#dce3e4",
        "text": "#222222",          # Black
    }

    # --- Google Fonts ---
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)


    # --- CSS Styles ---
    st.markdown(f"""
    <style>
    body, .stApp {{
        background-color: {theme['background']};
        font-family: 'Poppins', sans-serif;
        color: {theme['text']};
    }}
    .header {{
        background-color: {theme['primary']};
        padding: 15px;
        color: white;
        font-weight: bold;
        font-size: 26px;
        top: 0;
        width: 100%;
        z-index: 1000;
        display: flex;
        justify-content: center;
        align-items: center;
    }}


    /* Directly target the tab list (stronger selector) */
    div[data-baseweb="tab-list"] {{
        background-color: transparent !important;
        justify-content: center !important;
        box-shadow: none !important;
        border: none !important;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        justify-content: center;
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: {theme['primary2']};
        color: white;
        padding: 10px 16px;
        border-radius: 8px;
        font-weight: 600;
        transition: background-color 0.3s ease;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        background-color: {theme['background']};
    }}

    .stTabs [aria-selected="true"] {{
        background-color: {theme['background']};
        color: white;
    }}

.e10vaf9m1.st-emotion-cache-1f3w014.ex0cdmw0 {{
    color: black !important;
}}
    h2, h3, .main-title {{
        color: {theme['primary']};
        font-weight: 700;
    }}

    p, label, .stText, .stMarkdown {{
        color: {theme['text']};
    }}

    .stButton > button {{
        background-color: #A1CFDD !important;
        color: white !important;
        font-weight: 600;
    }}

    .st-emotion-cache-wbtvu4.egc9vxm0 {{
    background-color: transparent !important;
    box-shadow: none !important;
    border: none !important;
    padding: 0px !important;
    }}

    .st-emotion-cache-98xf22.ejh2rmr0 {{
    color: black !important;}}

    .st-emotion-cache-1q82h82.e14qm3311{{
    color: black !important;}}

    .st-af.st-ag.st-ah.st-ai.st-aj.st-ak.st-al.st-am.st-an.st-ao.st-ap.st-aq.st-ar {{
    background-color: #f5f5f5 !important;  /* Replace with your palette color */
    padding: 15px;
    border-radius: 12px;
    }}

    .stFileUploader > div > div {{
        background-color: white !important;
        border: 2px dashed #ccc !important;
        border-radius: 12px !important;
        padding: 20px !important;
    }}

    .centered-image {{
        text-align: center;
        margin: 30px 0;
    }}

    .centered-image img {{
        width: 400px;
        border-radius: 20px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        transition: transform 0.3s ease;
    }}

    .centered-image img:hover {{
        transform: scale(1.05);
    }}

    .quote {{
        color: {theme['primary']};
        font-style: italic;
        font-weight: 600;
        font-size: 20px;
        text-align: center;
        margin: 20px 0 40px 0;
    }}
    </style>
    """, unsafe_allow_html=True)


    def normalize_candidate_name(name: str) -> str:
        """Normalize candidate names to avoid duplicates and clean up."""
        if not name:
            return "Unknown Candidate"
        name = re.sub(r'(_CV|_cv|Cv|_Resume|Resume)$', '', name, flags=re.IGNORECASE)
        name = name.replace("_", " ").replace("-", " ")
        name = " ".join(name.split()).strip().title()
        return name

    # --- Initialize session state variables ---
    # These are crucial for storing processed data and caching results across reruns
    if "uploaded_cvs" not in st.session_state:
        st.session_state.uploaded_cvs = {} # Stores raw text content if needed for some modules
    if "candidate_cv_map" not in st.session_state:
        st.session_state.candidate_cv_map = {} # Normalized names to full text content
    if "all_documents" not in st.session_state:
        st.session_state.all_documents = [] # All Langchain Document chunks
    if "vector_store_initialized" not in st.session_state:
        st.session_state.vector_store_initialized = False # Flag for vector store status
    if "processing_status" not in st.session_state: # ADDED THIS LINE
        st.session_state.processing_status = "" # For user feedback during processing
    if "quick_recommendations_overview" not in st.session_state:
        st.session_state.quick_recommendations_overview = None
    if "job_recommendations" not in st.session_state: # Keep these if you use them in other tabs
        st.session_state.job_recommendations = None
    if "hr_questions_output" not in st.session_state: # Keep these if you use them in other tabs
        st.session_state.hr_questions_output = None
    if "cv_summaries" not in st.session_state: # Keep these if you use them in other tabs
        st.session_state.cv_summaries = {}


    # Initialize modules (using direct initialization as per your code)
    processor = CVProcessor(single_chunk=True)
    vector_store = CVVectorStore()
    rag_engine = EnhancedRAGEngine(vector_store, max_candidates_per_query=15)
    job_matcher = EnhancedJobMatcher(vector_store, rag_engine)
    summarizer = CVSummarizer()
    job_recommender = JobRecommender()
    hr_question_generator = HRQuestionGenerator()

    #st.set_page_config(page_title="Smart Recruiter Assistant", layout="wide")
    st.markdown(
        "<h1 style='text-align: center; color: #017691;'>Smart Recruiter Assistant</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center;'>Upload CVs, analyze them, ask queries, match jobs, and generate HR questions.</p>",
        unsafe_allow_html=True
    )
    # REMOVED DUPLICATE `if "uploaded_cvs" not in st.session_state:` BLOCK HERE

    st.subheader("Upload CVs")
    # Added a key to the file uploader for uniqueness
    uploaded_files = st.file_uploader("Upload CVs (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True, key="main_cv_uploader")

    if st.button("Process CVs"):
        if uploaded_files:
            st.session_state.processing_status = "Starting CV processing..."
            st.info(st.session_state.processing_status)

            with st.spinner("Processing CVs..."):
                try:
                    # Clear previous state on new processing
                    st.session_state.candidate_cv_map = {} # This will now store content by file.name
                    st.session_state.all_documents = [] # This stores all Langchain chunks for vector store
                    st.session_state.vector_store_initialized = False
                    st.session_state.quick_recommendations_overview = None
                    st.session_state.job_recommendations = None
                    st.session_state.hr_questions_output = None
                    st.session_state.cv_summaries = {}
                    st.session_state.uploaded_cvs = {} # Ensure this is cleared too if it's the raw text map

                    upload_dir = "uploaded_files"
                    os.makedirs(upload_dir, exist_ok=True)
                    
                    progress_bar = st.progress(0)
                    
                    processed_files_count = 0 # To track how many CV *files* were successfully processed

                    for i, file in enumerate(uploaded_files):
                        file_path = os.path.join(upload_dir, file.name)
                        
                        try:
                            with open(file_path, "wb") as f:
                                f.write(file.getbuffer())
                            
                            chunks = processor.process_cv(file_path)
                            # Use the original filename as the key for candidate_cv_map
                            # This ensures each uploaded file counts as a separate "candidate"
                            unique_candidate_key = file.name # Represents the unique CV file

                            if chunks:
                                st.session_state.uploaded_cvs[unique_candidate_key] = chunks[0].page_content 
                                
                                # Populate candidate_cv_map with a unique key per file
                                st.session_state.candidate_cv_map[unique_candidate_key] = chunks[0].page_content 
                                
                                for chunk in chunks:
                                    # Ensure metadata uses a consistent, unique identifier for the candidate
                                    chunk.metadata["candidate_name"] = unique_candidate_key 
                                    chunk.metadata["source_file"] = file.name
                                
                                st.session_state.all_documents.extend(chunks) # Add all chunks to all_documents
                                processed_files_count += 1 # Increment only for successfully processed files
                            else:
                                st.warning(f"No content extracted from '{file.name}'. It might be empty or unreadable.")

                        except Exception as e:
                            st.error(f"Error processing {file.name}: {str(e)}")
                            logger.error(f"Error processing {file.name}: {str(e)}", exc_info=True)
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    if st.session_state.all_documents:
                        vector_store.add_cvs(st.session_state.all_documents)
                        st.session_state.vector_store_initialized = True
                        # Use processed_files_count for the success message
                        st.success(f"✅ Successfully processed {processed_files_count} CV(s) and updated vector store.")
                    else:
                        st.warning("No valid CV content found after processing. Vector store not updated.")
                        st.session_state.vector_store_initialized = False

                except Exception as e:
                    st.error(f"An unexpected error occurred during processing: {str(e)}")
                    logger.error(f"Unexpected error during processing: {str(e)}", exc_info=True)
        else:
            st.warning("Please upload CV files first.")

    # TABS
    
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Ask", "Match", "Summarize", "Recommend", "HR Questions"])

    with tab0:
        st.header("Overview Dashboard")

        if not st.session_state.candidate_cv_map:
            pass
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Candidates", len(st.session_state.candidate_cv_map))
            
            with col2:
                # RENAMED METRIC: Now counts actual CV files processed
                st.metric("Candidates Processed", len(st.session_state.candidate_cv_map)) # Corrected to show count of unique CV files processed
            
            with col3:
                if st.session_state.vector_store_initialized:
                    st.metric("Vector Store", "✅ Ready")
                else:
                    st.metric("Vector Store", "❌ Not Ready")

            st.subheader("🎯 Quick Job Match Overview")
            
            generate_button_clicked = st.button("Generate Quick Job Overview", key="generate_overview_button_tab0")

            if generate_button_clicked:
                with st.spinner("Analyzing job matches... This might take a moment if many CVs."):
                    try:
                        quick_recommendations_raw = job_recommender.get_best_jobs_for_candidates(
                            st.session_state.candidate_cv_map, top_k=1
                        )
                        st.session_state.quick_recommendations_overview = quick_recommendations_raw
                        st.success("Job overview generated!")
                        
                    except Exception as e:
                        st.error(f"Error generating overview: {str(e)}")
                        st.session_state.quick_recommendations_overview = None
                        logger.error(f"Error generating quick overview: {str(e)}", exc_info=True)
            
            if st.session_state.quick_recommendations_overview:
                job_counts = Counter()
                for candidate_name, jobs_list in st.session_state.quick_recommendations_overview.items():
                    if jobs_list: 
                        top_job = jobs_list[0][0]  
                        job_counts[top_job] += 1
                
                if job_counts:
                    df = pd.DataFrame(list(job_counts.items()), columns=["Job Title", "Count"])
                    
                    fig = px.bar(df, x="Job Title", y="Count", 
                                    title="Distribution of Top Job Matches",
                                    color="Count", 
                                    labels={"Job Title": "Recommended Job Role", "Count": "Number of Candidates"},
                                    hover_data={"Job Title": True, "Count": True}) 

                    # Update x and y axis tick label color
                    fig.update_xaxes(tickfont=dict(color="black"))
                    fig.update_yaxes(tickfont=dict(color="black"))

                    fig.update_layout(
                        showlegend=False,
                        plot_bgcolor="rgba(220, 227, 228, 1)",
                        paper_bgcolor="rgba(220, 227, 228, 1)",
                        font=dict(color="black"),  
                        
                         # --- EDITED PLOT TITLE TO BE IN THE MIDDLE ---
                        title=dict(
                            text="Distribution of Candidates Across Top Job Roles", # Re-state the title text here
                            x=0.5,           # Center the title horizontally (0.5 means 50% of the plot width)
                            xanchor='center',# Anchor the title text itself from its center point
                            font=dict(
                                color="black", # <--- DEEPER RED IN RGBA
                                size=24 # <--- INCREASED FONT SIZE (adjust as needed, e.g., 28, 30)
                            )

                        ),

                        xaxis_title_font_color="black", 
                        yaxis_title_font_color="black", 
                        
                        coloraxis_colorbar=dict(
                            title=dict(
                                text="Number of Candidates",
                                font=dict(color="black")
                            )
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True) 
                    
                    st.subheader("🏆 Top Job Categories")
                    for job, count in job_counts.most_common():
                        st.write(f"- **{job}**: {count} candidates")
                else:
                    st.info("No job matches found for the processed CVs. Try uploading more diverse CVs or check job definitions.")
            elif not generate_button_clicked and st.session_state.quick_recommendations_overview is None:
                st.info("Click 'Generate Quick Job Overview' to see the distribution of candidates across top job categories.")

    with tab1:
        st.subheader("Ask a question")
        query = st.text_input("Enter your query")

        if st.button("Run Query") and query.strip():
            try:
                top_text = rag_engine.find_top_candidates(query, top_k=5)
                all_relevant = rag_engine.get_all_candidates_for_skill(query)
                st.subheader("🎯 Top Candidates")
                st.code(top_text[0], language="markdown")
            except Exception as e:
                st.error(f"❌ Failed to process query: {e}")

    with tab2:
        st.subheader("Match job description")
        job_desc = st.text_area("Paste job description")
        if st.button("Match"):
            if st.session_state.uploaded_cvs:
                results = job_matcher.match_job_to_cvs(job_desc, top_k=5, explain=True)
                for res in job_matcher.format_results(results).split("\n"):
                    st.markdown(res)
            else:
                st.warning("Upload and process CVs first.")

    with tab3:
        st.subheader("Summarize CVs")
        if st.button("Summarize"):
            for name, content in st.session_state.uploaded_cvs.items():
                summary = summarizer.summarize_cv(content, name)
                st.markdown(f"**{name}**")
                st.success(summary)

    with tab4:
        st.subheader("Job Recommendations")
        if st.button("Recommend Jobs"):
            recommendations = job_recommender.get_best_jobs_for_candidates(st.session_state.uploaded_cvs, top_k=3)
            for job, candidates in recommendations.items():
                st.markdown(f"### {job}")
                for candidate, score, reason in candidates:
                    st.markdown(f"**Candidate:** {candidate}  ")
                    st.markdown(f"**Score:** {score:.2f}  ")
                    st.markdown(f"**Reason:** {reason}")
                    st.markdown("---")

    with tab5:
        st.subheader("HR Interview Questions")
        if st.button("Generate Questions"):
            if st.session_state.uploaded_cvs:
                top_names = list(st.session_state.uploaded_cvs.keys())[:5]
                questions = hr_question_generator.generate_questions_for_top_candidates(st.session_state.uploaded_cvs, top_names)
                questions = hr_question_generator.generate_questions_for_top_candidates(st.session_state.uploaded_cvs, top_names)
                for name in top_names:
                    st.markdown(f"### {name}")
                    for sec, qs in questions[name].items():
                        st.markdown(f"**{sec}**")
                        for q in qs:
                            st.markdown(f"- {q}")
            else:
                st.warning("Upload and process CVs first.")


