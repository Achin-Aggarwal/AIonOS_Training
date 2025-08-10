import json
import pandas as pd
from pathlib import Path
import os
import tempfile
import shutil
from typing import List, Optional, Tuple
import hashlib

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("✅ Using updated HuggingFaceEmbeddings")
except ImportError:
    from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings as HuggingFaceEmbeddings
    print("⚠️ Using deprecated SentenceTransformerEmbeddings (consider upgrading)")

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

PDF_FOLDER = "Resume"  
DB_FOLDER = "./resume_db"  
MODEL_NAME = "gemma3:1b"  
COLLECTION_NAME = "resume-collection"
EMBEDDING_MODEL = "thenlper/gte-large"
TEMP_DB_PREFIX = "temp_resume_db_"

def create_pdf_hash(pdf_files: List) -> str:
    if not pdf_files:
        return "empty"
    
    hash_string = ""
    for pdf_file in pdf_files:
        if hasattr(pdf_file, 'name') and hasattr(pdf_file, 'size'):
            hash_string += f"{pdf_file.name}_{pdf_file.size}_"
    
    return hashlib.md5(hash_string.encode()).hexdigest()[:10]

def save_uploaded_pdfs(uploaded_files: List, temp_dir: str) -> List[str]:
    saved_paths = []
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        saved_paths.append(file_path)
    
    return saved_paths

def process_uploaded_pdfs(uploaded_files: List) -> Tuple[Optional[str], Optional[List[Document]], Optional[List[str]]]:
    if not uploaded_files:
        return None, None, None
    
    try:
        temp_dir = tempfile.mkdtemp(prefix=TEMP_DB_PREFIX)
        saved_paths = save_uploaded_pdfs(uploaded_files, temp_dir)
        
        documents = []
        filenames = []
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name='cl100k_base',
            chunk_size=512,
            chunk_overlap=16
        )
        
        for file_path in saved_paths:
            filename = os.path.basename(file_path)
            filenames.append(filename)
            loader = PyPDFLoader(file_path)
            file_docs = loader.load_and_split(text_splitter)
            
            for doc in file_docs:
                doc.metadata['source_filename'] = filename
                doc.metadata['original_source'] = filename
            
            documents.extend(file_docs)
        
        print(f"📊 Processed {len(documents)} document chunks from {len(uploaded_files)} PDFs")
        return temp_dir, documents, filenames
        
    except Exception as e:
        print(f"❌ Error processing uploaded PDFs: {e}")
        return None, None, None

def create_dynamic_vectorstore(documents: List[Document], session_id: str) -> Optional[Chroma]:
    if not documents:
        return None
    
    try:
        try:
            embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-large")
            print(f"✅ Created embedding model: thenlper/gte-large")
        except Exception as e:
            print(f"⚠️ Error with thenlper/gte-large, trying fallback model...")
            embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            print("✅ Using fallback embedding model: all-MiniLM-L6-v2")
        
        temp_db_dir = f"./temp_dbs/session_{session_id}"
        os.makedirs(temp_db_dir, exist_ok=True)
        
        vectorstore = Chroma.from_documents(
            documents,
            embedding_model,
            collection_name=f"session_{session_id}",
            persist_directory=temp_db_dir
        )
        vectorstore.persist()
        
        print(f"✅ Created temporary vector database for session {session_id}")
        return vectorstore
        
    except Exception as e:
        print(f"❌ Error creating dynamic vectorstore: {e}")
        print("💡 Make sure you have installed: pip install sentence-transformers langchain-huggingface")
        return None

def cleanup_temp_files(temp_dir: str, temp_db_dir: str = None):
    try:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"🗑️ Cleaned up temp PDF directory: {temp_dir}")
        
        if temp_db_dir and os.path.exists(temp_db_dir):
            shutil.rmtree(temp_db_dir)
            print(f"🗑️ Cleaned up temp DB directory: {temp_db_dir}")
            
    except Exception as e:
        print(f"⚠️ Error cleaning up temp files: {e}")

def setup_document_processing():
    print("📄 Setting up document processing...")
    
    if not Path(PDF_FOLDER).exists():
        print(f"❌ PDF folder '{PDF_FOLDER}' not found!")
        print("Please create the folder and add your PDF files.")
        return None, None
    
    pdf_loader = PyPDFDirectoryLoader(PDF_FOLDER)
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name='cl100k_base',
        chunk_size=512,
        chunk_overlap=16
    )
    
    print("✅ Document processing setup complete")
    return pdf_loader, text_splitter

def create_vector_database(pdf_loader, text_splitter):
    print("🔄 Creating vector database...")
    
    try:
        resume_chunks = pdf_loader.load_and_split(text_splitter)
        print(f"📊 Loaded {len(resume_chunks)} document chunks")
        
        try:
            embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-large")
            print(f"✅ Created embedding model: thenlper/gte-large")
        except Exception as e:
            print(f"⚠️ Error with thenlper/gte-large, trying fallback model...")
            embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            print("✅ Using fallback embedding model: all-MiniLM-L6-v2")
        
        vectorstore = Chroma.from_documents(
            resume_chunks,
            embedding_model,
            collection_name=COLLECTION_NAME,
            persist_directory=DB_FOLDER
        )
        vectorstore.persist()
        
        print("✅ Vector database created and persisted")
        return embedding_model, vectorstore
        
    except Exception as e:
        print(f"❌ Error creating vector database: {e}")
        print("💡 Make sure you have installed: pip install sentence-transformers langchain-huggingface")
        return None, None

def load_vector_database():
    print("🔄 Loading existing vector database...")
    
    try:
        try:
            embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-large")
            print(f"✅ Created embedding model: thenlper/gte-large")
        except Exception as e:
            print(f"⚠️ Error with thenlper/gte-large, trying fallback model...")
            embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            print("✅ Using fallback embedding model: all-MiniLM-L6-v2")
        
        vectorstore_persisted = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=DB_FOLDER,
            embedding_function=embedding_model
        )
        
        print("✅ Vector database loaded successfully")
        return embedding_model, vectorstore_persisted
        
    except Exception as e:
        print(f"❌ Error loading vector database: {e}")
        print("💡 Make sure you have installed: pip install sentence-transformers langchain-huggingface")
        return None, None

def initialize_ollama():
    print(f"🤖 Initializing Ollama with model: {MODEL_NAME}")
    
    try:
        llm = Ollama(model=MODEL_NAME)
        test_response = llm.invoke("Hello")
        print("✅ Ollama initialized successfully")
        return llm
        
    except Exception as e:
        print(f"❌ Error initializing Ollama: {e}")
        print("💡 Make sure Ollama is running (ollama serve) and the model is pulled.")
        print(f"   Try: ollama pull {MODEL_NAME}")
        return None

qna_system_message = """
You are an expert document analysis assistant with advanced capabilities in extracting, synthesizing, and presenting information from PDF documents. Your primary functions include comprehensive document summarization, precise question answering, and educational content generation.

CORE COMPETENCIES:
- Advanced text comprehension and synthesis
- Multi-document comparative analysis
- Structured information extraction
- Educational content creation
- Professional document assessment

OPERATIONAL GUIDELINES:

1. **ENHANCED SUMMARIZATION MODE**
   When users request summaries, provide in-depth, professionally structured analysis:
   
   For EACH document, deliver:
   ### Document: [ORIGINAL_FILENAME]
   
   **Executive Summary:** [2-3 sentence overview]
   **Key Highlights:** [Main points, achievements, qualifications]
   **Technical Details:** [Specific skills, technologies, certifications]
   **Professional Background:** [Experience levels, industries, roles]
   **Notable Elements:** [Unique aspects, standout features]
   **Assessment:** [Professional evaluation of strengths]
   
   Requirements:
   - Extract ALL significant information from each document
   - Maintain professional, analytical tone
   - Provide actionable insights
   - Ensure completeness and accuracy
   - Use clear, structured formatting

2. **PRECISION QUESTION ANSWERING**
   Deliver comprehensive, evidence-based responses:
   - Analyze ALL relevant context thoroughly
   - Provide detailed, multi-faceted answers
   - Include specific examples and evidence
   - Cross-reference information when applicable
   - Maintain factual accuracy and completeness
   - Structure complex answers with clear organization

3. **ADVANCED QUIZ GENERATION**
   Create sophisticated assessment materials:
   - Generate minimum 12-15 high-quality questions
   - Employ diverse question types:
     * Multiple choice (4 options, single correct)
     * True/False with justification requirements
     * Short answer (2-3 sentences expected)
     * Fill-in-the-blank with context
     * Scenario-based application questions
     * Comparative analysis questions
   - Ensure comprehensive content coverage
   - Include questions of varying difficulty levels
   - Focus on practical application and understanding

4. **MULTI-REQUEST PROCESSING**
   Handle complex, multi-part requests systematically:
   - Parse all request components accurately
   - Execute each task with full attention to detail
   - Provide clear section delineation
   - Maintain consistent quality across all deliverables

QUALITY STANDARDS:
- Professional, clear, and engaging communication
- Comprehensive coverage of all available information
- Accurate representation of document contents
- Structured, logical presentation
- Evidence-based analysis and recommendations

CONSTRAINTS:
- Rely exclusively on provided context information
- Never fabricate or assume information not present
- Maintain objectivity and factual accuracy
- Preserve original document integrity and meaning

Your responses should demonstrate expertise, thoroughness, and professional insight while remaining accessible and actionable for the user.
"""

qna_user_message_template = """
###Context
Here are the relevant document sections for analysis:
{context}

###Question
{question}
"""

def get_original_filenames_from_context(relevant_documents: List[Document]) -> List[str]:
    filenames = set()
    for doc in relevant_documents:
        if 'source_filename' in doc.metadata:
            filenames.add(doc.metadata['source_filename'])
        elif 'original_source' in doc.metadata:
            filenames.add(doc.metadata['original_source'])
        elif 'source' in doc.metadata:
            source_path = doc.metadata['source']
            filename = os.path.basename(source_path)
            filenames.add(filename)
    return list(filenames)

def run_qna_pipeline(user_input, retriever, qna_system_message, qna_user_message_template, llm):
    try:
        relevant_document_chunks = retriever.get_relevant_documents(user_input)
        
        filenames = get_original_filenames_from_context(relevant_document_chunks)
        
        context_list = []
        for doc in relevant_document_chunks:
            filename = "Unknown"
            if 'source_filename' in doc.metadata:
                filename = doc.metadata['source_filename']
            elif 'original_source' in doc.metadata:
                filename = doc.metadata['original_source']
            elif 'source' in doc.metadata:
                filename = os.path.basename(doc.metadata['source'])
            
            context_entry = f"[Source: {filename}]\n{doc.page_content}"
            context_list.append(context_entry)
        
        context_for_query = "\n\n".join(context_list)
        
        if "summary" in user_input.lower() or "summarize" in user_input.lower():
            enhanced_context = f"Available documents for analysis: {', '.join(filenames)}\n\n{context_for_query}"
        else:
            enhanced_context = context_for_query

        full_prompt = f"{qna_system_message}\n\n{qna_user_message_template.format(context=enhanced_context, question=user_input)}"

        response = llm.invoke(full_prompt)

        return response.strip()

    except Exception as e:
        return f"Sorry, I encountered the following error: \n{e}"

def setup_dynamic_system(uploaded_files: List = None, session_id: str = "default"):
    print("🚀 Starting Dynamic Resume RAG System Setup...")
    print("=" * 50)
    
    vectorstore_persisted = None
    temp_dir = None
    filenames = None
    
    if uploaded_files:
        print(f"📁 Processing {len(uploaded_files)} uploaded PDF files...")
        
        temp_dir, documents, filenames = process_uploaded_pdfs(uploaded_files)
        if documents:
            vectorstore_persisted = create_dynamic_vectorstore(documents, session_id)
        else:
            print("❌ Failed to process uploaded PDFs")
            return None, None, None, None, None
    else:
        print("📁 No uploaded files, checking for default Resume folder...")
        db_exists = Path(DB_FOLDER).exists()
        
        if db_exists:
            print("📁 Existing database found, loading...")
            embedding_model, vectorstore_persisted = load_vector_database()
        else:
            print("📁 No existing database found, creating new one...")
            pdf_loader, text_splitter = setup_document_processing()
            if pdf_loader is None:
                print("❌ No PDFs found in default folder and no files uploaded")
                return None, None, None, None, None
            
            embedding_model, vectorstore_persisted = create_vector_database(pdf_loader, text_splitter)
    
    if vectorstore_persisted is None:
        print("❌ Failed to setup vector database")
        return None, None, None, None, None
    
    llm = initialize_ollama()
    if llm is None:
        return None, None, None, None, None
    
    retriever = vectorstore_persisted.as_retriever(
        search_type='similarity',
        search_kwargs={'k': 8}
    )
    
    print("=" * 50)
    print("✅ Resume RAG System setup complete!")
    print(f"🤖 Model: {MODEL_NAME}")
    print(f"🔍 Retriever: k=8 similarity search")
    if uploaded_files:
        print(f"📊 Processing {len(uploaded_files)} uploaded PDFs")
    else:
        print(f"📊 Database location: {DB_FOLDER}")
    
    return llm, retriever, vectorstore_persisted, temp_dir, filenames

def setup_system():
    llm, retriever, vectorstore_persisted, _, _ = setup_dynamic_system()
    return llm, retriever, vectorstore_persisted

if __name__ == "__main__":
    llm, retriever, vectorstore_persisted = setup_system()
    
    if llm is None:
        print("❌ System setup failed. Please check the errors above.")
        print("\n💡 Common solutions:")
        print("1. Install missing packages: pip install sentence-transformers langchain-huggingface")
        print("2. Make sure Ollama is running: ollama serve")
        print("3. Pull the model: ollama pull gemma3:1b")
        exit(1)
    
    print("✅ Resume RAG System ready!")
    print("🚀 To start the Streamlit app, run:")
    print("streamlit run streamlit_app.py")

try:
    if 'llm' not in globals():
        llm, retriever, vectorstore_persisted = setup_system()
except:
    llm = retriever = vectorstore_persisted = None