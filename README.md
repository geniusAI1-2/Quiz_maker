# Quiz_maker
This FastAPI application provides a comprehensive solution for processing educational content from various sources (YouTube, PDFs, voice instructions) and generating educational materials like questions and assessments.

## Features

- **YouTube Transcript Processing**: Extract and store transcripts from YouTube videos
- **PDF Content Processing**: Upload and index PDF documents for educational content
- **Question Generation**: Create various types of questions (multiple choice, true/false, short answer) from indexed content
- **Answer Evaluation**: Evaluate student answers against correct answers
- **Voice Instruction Processing**: Convert voice instructions to educational content
- **Math & Physics Solver**: Solve math/physics problems with step-by-step explanations and visualizations

## API Endpoints

### 1. YouTube Transcript Processing
- **Endpoint**: `/getting_script`
- **Method**: POST
- **Input**: YouTube URL, language, index name
- **Output**: Stores transcript in FAISS vector store

### 2. PDF Processing
- **Endpoint**: `/upload_pdf`
- **Method**: POST
- **Input**: PDF file, index name
- **Output**: Processes and indexes PDF content

### 3. Question Generation
- **Endpoint**: `/generate_questions`
- **Method**: POST
- **Input**: Subject, number of questions, index name, question type
- **Output**: Generated questions with answers

### 4. Answer Evaluation
- **Endpoint**: `/evaluation`
- **Method**: POST
- **Input**: Questions, student answers, correct answers
- **Output**: Detailed evaluation results

### 5. Voice Instruction Processing
- **Endpoint**: `/voice_script`
- **Method**: POST
- **Input**: Audio file, index name
- **Output**: Generated educational content from voice instruction

### 6. Math & Physics Solver
- **Endpoint**: `/math&physics`
- **Method**: POST
- **Input**: Math/physics question
- **Output**: Solution with explanation and visualization code

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/educational-content-processor.git
   cd educational-content-processor
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment variables**:
   Create a `.env` file with the following variables:
   ```
   GROQ_APIKEY=your_groq_api_key
   GOOGLE_API_KEY=your_google_api_key
   ```

4. **Run the application**:
   ```bash
   uvicorn main:app --reload
   ```

## Dependencies

- FastAPI
- Python-dotenv
- YouTube-transcript-api
- Groq
- Langchain
- FAISS
- HuggingFace embeddings
- Google Generative AI

## Usage Examples

### YouTube Transcript Processing
```python
import requests

url = "http://localhost:8000/getting_script"
data = {
    "input_link": "https://www.youtube.com/watch?v=example",
    "language": "en",
    "index_name": "physics_lectures"
}
response = requests.post(url, json=data)
```

### PDF Processing
```python
import requests

url = "http://localhost:8000/upload_pdf"
files = {'file': open('lecture.pdf', 'rb')}
data = {'index_name': 'physics_lectures'}
response = requests.post(url, files=files, data=data)
```

### Question Generation
```python
import requests

url = "http://localhost:8000/generate_questions"
data = {
    "subject": "quantum mechanics",
    "num_questions": 5,
    "index_name": "physics_lectures",
    "question_type": "multiple_choice"
}
response = requests.post(url, json=data)
```
