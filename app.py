from flask import Flask, render_template, request
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import docx2txt
import PyPDF2

app = Flask(__name__)

# App Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'docx', 'pdf'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure NLTK Data Path
NLTK_DATA_PATH = os.path.join(app.root_path, 'static', 'nltk_data')
nltk.data.path.append(NLTK_DATA_PATH)

# Ensure required NLTK datasets are downloaded
nltk.download('punkt', download_dir=NLTK_DATA_PATH)
nltk.download('punkt_tab', download_dir=NLTK_DATA_PATH)
nltk.download('stopwords', download_dir=NLTK_DATA_PATH)

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_text(text):
    """Preprocess the text by tokenizing, removing stopwords, and applying stemming."""
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokens = word_tokenize(text.lower())
    filtered_tokens = [stemmer.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(filtered_tokens)

def read_docx(file):
    """Read content from a .docx file."""
    return docx2txt.process(file)

def read_pdf(file):
    """Read content from a .pdf file."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def read_file(file):
    """Read file content based on its extension."""
    if file.filename.endswith('.txt'):
        return file.read().decode("utf-8")
    elif file.filename.endswith('.docx'):
        return read_docx(file)
    elif file.filename.endswith('.pdf'):
        return read_pdf(file)
    return ""  # Return empty string if unsupported file format

def calculate_similarity(doc1, doc2):
    """Calculate and return the percentage similarity between two preprocessed documents."""
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([doc1, doc2])
    similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarities[0][1] * 100

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Retrieve uploaded files
        file1 = request.files.get('file1')
        file2 = request.files.get('file2')

        if not file1 or not file2:
            return render_template("index.html", similarity=None, error="Both files are required.")

        if file1.filename == '' or file2.filename == '':
            return render_template("index.html", similarity=None, error="No file selected.")

        if allowed_file(file1.filename) and allowed_file(file2.filename):
            try:
                # Read and preprocess file content
                file1_content = read_file(file1)
                file2_content = read_file(file2)

                if not file1_content or not file2_content:
                    return render_template("index.html", similarity=None, error="Error reading files.")

                preprocessed_doc1 = preprocess_text(file1_content)
                preprocessed_doc2 = preprocess_text(file2_content)

                # Calculate similarity
                similarity_percentage = calculate_similarity(preprocessed_doc1, preprocessed_doc2)
                return render_template("index.html", 
                                       similarity=f"Percentage similarity between {file1.filename} and {file2.filename}: {similarity_percentage:.2f}%", 
                                       error=None)

            except Exception as e:
                return render_template("index.html", similarity=None, error=f"An error occurred: {str(e)}")
        else:
            return render_template("index.html", similarity=None, error="Invalid file format. Please upload .txt, .docx, or .pdf files.")

    return render_template("index.html", similarity=None, error=None)

if __name__ == "__main__":
    app.run(debug=True)
