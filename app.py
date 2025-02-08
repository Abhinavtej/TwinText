import streamlit as st
import nltk
import docx2txt
import PyPDF2
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure required NLTK datasets are downloaded
nltk.download("punkt")
nltk.download("stopwords")
nltk.download('punkt_tab')

# Function to preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    tokens = word_tokenize(text.lower())
    filtered_tokens = [stemmer.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
    return " ".join(filtered_tokens)

# Function to read .docx file
def read_docx(file):
    return docx2txt.process(file)

# Function to read .pdf file
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Function to read file based on its type
def read_file(file, file_type):
    if file_type == "txt":
        return file.getvalue().decode("utf-8")
    elif file_type == "docx":
        return read_docx(file)
    elif file_type == "pdf":
        return read_pdf(file)
    return ""

# Function to calculate similarity
def calculate_similarity(doc1, doc2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)[0][1] * 100
    return similarity

# Streamlit UI
st.set_page_config(page_title="Document Similarity Checker", layout="centered")
st.title("📄 Document Similarity Checker")

st.markdown("Upload two documents (.txt, .docx, .pdf) to compare their similarity.")

# File upload
file1 = st.file_uploader("Upload Document 1", type=["txt", "docx", "pdf"])
file2 = st.file_uploader("Upload Document 2", type=["txt", "docx", "pdf"])

if file1 and file2:
    try:
        file1_content = read_file(file1, file1.type.split("/")[-1])
        file2_content = read_file(file2, file2.type.split("/")[-1])

        if file1_content and file2_content:
            preprocessed_doc1 = preprocess_text(file1_content)
            preprocessed_doc2 = preprocess_text(file2_content)

            similarity_score = calculate_similarity(preprocessed_doc1, preprocessed_doc2)
            st.success(f"✅ Similarity: {similarity_score:.2f}%")
        else:
            st.error("Error reading one or both files.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


st.markdown("Developed by [Abhinavtej Reddy](https://abhinavtejreddy.me)")