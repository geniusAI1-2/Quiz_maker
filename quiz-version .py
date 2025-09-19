from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import os
import uuid
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import re
from groq import Groq
from typing import Optional,Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import subprocess
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import uvicorn
from openai import OpenAI  # or other lib you're using for Whisper
import tempfile

load_dotenv()

app = FastAPI()

DEFAULT_FAISS_INDEX = "faiss_index"


BASE_FAISS_DIR = "faiss_indexes" 
os.makedirs(BASE_FAISS_DIR, exist_ok=True)

# Initialize Google Gemini
gemini_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7
)

# Initialize Groq S2T
Groq_model = Groq(api_key=(os.getenv("GROQ_APIKEY")))


# Initialize embeddings
hf_embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Request models
class GettingYouTubeScript(BaseModel):
    input_link: str
    language: str
    index_name: str

class MakingQuestions(BaseModel):
    subject: str  
    num_questions: int
    index_name: str
    question_type: str

class UploadResponse(BaseModel):
    message: str
    index_path: Optional[str] = None

class EvaluateTheAnswers(BaseModel):
    questions: list[str]
    student_answers: list[str]
    correct_answers: list[str]

class MakingScript(BaseModel):
    index_name:str

class Question(BaseModel):
    input_Q: str
# Helper functions
def extract_video_id(url):
    if 'youtu.be' in url:
        return url.split('/')[-1].split('?')[0]
    query = urlparse(url).query
    params = parse_qs(query)
    return params.get('v', [None])[0]

def clean_response(text):
    return re.sub(r"[ﭼ-ﯿ]+", "", text)

def get_faiss_path(index_name: str) -> str:
    """دالة موحدة للحصول على المسار"""
    # تنظيف اسم الفهرس من المسارات
    base_name = os.path.basename(index_name)
    path = os.path.join(BASE_FAISS_DIR, base_name)
    os.makedirs(path, exist_ok=True)
    return path

def validate_faiss_index(index_path: str) -> bool:
    """التحقق من وجود الملفات الأساسية"""
    required_files = ['index.faiss', 'index.pkl']
    return all(os.path.exists(os.path.join(index_path, f)) for f in required_files)

def load_vectorstore(index_path: str):
    if validate_faiss_index(index_path):
        try:
            return FAISS.load_local(index_path, hf_embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Failed to load FAISS index at {index_path}: {e}")
            return None
    print(f"Index files missing or incomplete at {index_path}")
    return None

def save_vectorstore(vs, index_path: str):
    try:
        vs.save_local(index_path)
        print(f"Index saved successfully at {index_path}")
    except Exception as e:
        print(f"Failed to save index at {index_path}: {e}")
        raise

def add_text_to_vectorstore(text: str, metadata: dict, index_path: str):
    if not text:
        return
    
    vs = load_vectorstore(index_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    splits = text_splitter.create_documents([text], [metadata])

    if vs is None:
        vs = FAISS.from_documents(splits, hf_embeddings)
    else:
        vs.add_documents(splits)
    
    save_vectorstore(vs, index_path)

def add_docs_to_vectorstore(docs, index_path: str):
    vs = load_vectorstore(index_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    splits = text_splitter.split_documents(docs)

    if vs is None:
        vs = FAISS.from_documents(splits, hf_embeddings)
    else:
        vs.add_documents(splits)

    save_vectorstore(vs, index_path)

def normalize_question_type(raw_type: str) -> str:
    raw_type = raw_type.strip().lower()
    if any(word in raw_type for word in ["اختيار", "اختياري", "متعدد", "mcq", "multiple"]):
        return "multiple_choice"
    elif any(word in raw_type for word in ["صح", "خطأ", "true", "false"]):
        return "true_false"
    elif any(word in raw_type for word in ["قصيرة", "مفتوحة", "short", "open"]):
        return "short_answer"
    else:
        return "general"
    
def turn_S2T(file: UploadFile) -> str:


    try:
        client = OpenAI(api_key=os.getenv("GROQ_APIKEY"))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(file.file.read())
            tmp_path = tmp_file.name

        transcription = client.audio.transcriptions.create(
            file=open(tmp_path, "rb"),
            model="whisper-large-v3",
            response_format="text"
        )
        return transcription
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {str(e)}")


# Endpoints
@app.post("/quiz/getting_script")
async def get_script(request: GettingYouTubeScript):
    youtube_url = request.input_link
    language = request.language
    index_name = request.index_name

    def get_youtube_text(url, lang):
        try:
            video_id = extract_video_id(url)
            if not video_id:
                raise ValueError("Could not extract video ID from URL")

            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
            except:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])

            return " ".join([entry['text'] for entry in transcript])

        except Exception as e:
            print(f"Error getting YouTube transcript: {e}")
            return None

    text = get_youtube_text(youtube_url, language)
    index_path = get_faiss_path(index_name)

    if text:
        metadata = {
            "source": "youtube",
            "url": youtube_url,
            "language": language
        }
        add_text_to_vectorstore(text, metadata, index_path)
        
        return {
            "status": "success",
            "video_url": youtube_url,
            "language": language,
            "transcript": text,
            "index_path": index_path,
            "message": "تمت إضافة النص إلى قاعدة البيانات"
        }
    else:
        raise HTTPException(status_code=400, detail="فشل في الحصول على النص من رابط اليوتيوب.")



@app.post("/quiz/upload_pdf")
async def upload_pdf(file: UploadFile = File(...), index_name: str = Form(...)):
    try:
        temp_pdf = f"temp_{uuid.uuid4().hex}.pdf"
        with open(temp_pdf, "wb") as f:
            f.write(await file.read())
        
        loader = PyPDFLoader(temp_pdf)
        docs = loader.load()
        
        index_path = get_faiss_path(index_name)  # استخدام الدالة الموحدة
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
        splits = text_splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(splits, hf_embeddings)
        save_vectorstore(vectorstore, index_path)

        os.remove(temp_pdf)

        return UploadResponse(
            message="تم معالجة الملف وإضافته إلى قاعدة البيانات بنجاح",
            index_path=index_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quiz/generate_questions")
async def generate_questions(request: MakingQuestions):
    try:
        index_path = get_faiss_path(request.index_name)
        print(f"المسار المستخدم: {index_path}")  
        
        vs = load_vectorstore(index_path)
        if vs is None:
            available_indices = [d for d in os.listdir(BASE_FAISS_DIR) 
                              if os.path.isdir(os.path.join(BASE_FAISS_DIR, d))]
            raise HTTPException(
                status_code=400,
                detail=f"لا توجد بيانات متاحة. الفهرس المطلوب: {request.index_name}. الفهارس المتاحة: {available_indices}"
            )
        
        retriever = vs.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.get_relevant_documents(request.subject)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        normalized_type = normalize_question_type(request.question_type)

        # Set format instructions and examples based on normalized type
        if normalized_type == "multiple_choice":
            formatting_rules = """
        - Write each question on a new line.
        - Provide 3 or 4 answer choices inside square brackets: [Choice1, Choice2, Choice3].
        - The correct answer must be added after the question using: || Correct Answer.
        """
            example_output = """
        What is the capital of France? [London, Paris, Rome, Madrid] || Paris
        Which planet is known as the Red Planet? [Earth, Mars, Jupiter] || Mars
        """
        elif normalized_type == "true_false":
            formatting_rules = """
        - Write each statement on a new line.
        - Add the correct answer after the statement using: || True or False.
        """
            example_output = """
        The Sun is a planet. || False
        Water boils at 100°C. || True
        """
        elif normalized_type == "short_answer":
            formatting_rules = """
        - Write each question on a new line.
        - Add the correct answer after the question using: || Correct Answer.
        """
            example_output = """
        What is the chemical symbol for water? || H2O
        Who discovered gravity? || Isaac Newton
        """
        else:
            formatting_rules = """
        - Write each question on a new line.
        - Add the correct answer after the question using: || Correct Answer.
        """
            example_output = """
        What is the result of 5 + 7? || 12
        Define photosynthesis. || The process by which green plants convert sunlight into energy.
        """

        # Final dynamic prompt
        prompt = f"""
You are a professional teacher and expert in educational assessment. Your task is to generate {request.num_questions} questions in the subject of "{request.subject}", using the question type: "{request.question_type}", based only on the reference content provided below.

**Formatting Instructions based on Question Type:**
{formatting_rules}

**General Rules:**
1. Do not use any numbering, bullet points, or extra formatting.
2. Only use the reference content. Do not invent unrelated facts.
3. The correct answer must appear after ||.
4. Keep output clean and in plain text with no extra explanation.

**Reference Content:**
{context}

**Example Output:**
{example_output}
"""
        response = gemini_model.invoke(prompt)
        output = getattr(response, 'content', "").strip()

        questions = []
        answers = []
        
        for line in output.split('\n'):
            line = line.strip()
            if '||' in line:
                q, a = line.split('||', 1)
                questions.append(q.strip())
                answers.append(a.strip())

        if len(questions) != request.num_questions:
            raise HTTPException(
                status_code=500,
                detail=f"تم إنشاء {len(questions)} أسئلة فقط من {request.num_questions} المطلوبة"
            )

        return {
            "questions": questions,
            "answers": answers
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"خطأ في إنشاء الأسئلة: {str(e)}"
        )


@app.post("/quiz/evaluation")
async def evaluation(request: EvaluateTheAnswers):
    if not (len(request.questions) == len(request.student_answers) == len(request.correct_answers)):
        raise HTTPException(
            status_code=400,
            detail="يجب أن يكون عدد الأسئلة والإجابات متطابقاً"
        )
    
    try:
        score = 0
        detailed_results = []
        
        for i, (question, student_answer, correct_answer) in enumerate(zip(
            request.questions,
            request.student_answers,
            request.correct_answers
        )):
            is_correct = student_answer.strip().lower() == correct_answer.strip().lower()
            if is_correct:
                score += 1
            
            detailed_results.append({
                "question_number": i + 1,
                "question": question,
                "student_answer": student_answer,
                "correct_answer": correct_answer,
                "is_correct": is_correct
            })

        total_questions = len(request.questions)
        percentage = (score / total_questions) * 100 if total_questions > 0 else 0
        
        return {
            "score": score,
            "total_questions": total_questions,
            "percentage": f"{percentage:.2f}%",
            "detailed_results": detailed_results,
            "summary": f"النتيجة: {score}/{total_questions} ({percentage:.2f}%)"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"حدث خطأ أثناء التقييم: {str(e)}"
        )

@app.post("/quiz/voice_script")
async def voice_script_endpoint(
    file: UploadFile = File(...),
    index_name: str = Form(...)
):
    try:
        temp_audio_path = f"temp_instruction_{uuid.uuid4().hex}.mp3"
        with open(temp_audio_path, "wb") as f:
            f.write(await file.read())

        with open(temp_audio_path, "rb") as audio_file:
            transcription = Groq_model.audio.transcriptions.create(
                file=("audio.mp3", audio_file.read()),
                model="whisper-large-v3",
                response_format="verbose_json"
            )
        os.remove(temp_audio_path)
        instruction_text = transcription.text.strip()

        prompt = f"""
You are a professional educator and content creator.

Based on the following user instruction (spoken in the audio), generate an **original educational subject** suitable for students. The content should be structured clearly and be informative.

### Instruction from user (transcribed audio):
\"\"\"{instruction_text}\"\"\"

### Your Task:
Generate a full educational subject (lecture, summary, or lesson) based on that instruction. Be creative, but make sure it is coherent and valuable to learners.

Return only clean, plain text content.
"""

        response = gemini_model.invoke(prompt)
        generated_subject = getattr(response, 'content', "").strip()

        index_path = get_faiss_path(index_name)
        metadata = {
            "source": "voice_instruction",
            "instruction_text": instruction_text
        }
        add_text_to_vectorstore(generated_subject, metadata, index_path)

        return {
            "status": "success",
            "instruction_transcribed": instruction_text,
            "generated_subject_summary": generated_subject[:300] + "...",
            "index_path": index_path,
            "message": "تمت معالجة التعليمات الصوتية وإنشاء موضوع تعليمي جديد وإضافته إلى قاعدة البيانات."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"حدث خطأ: {str(e)}")


@app.post("/quiz/math&physics")
async def math_physics_endpoint(data: Question) -> Dict:
    prompt = f"""
أنت مساعد ذكي لحل مسائل الرياضيات والفيزياء.

مهمتك:

1. حل المسألة بشكل دقيق وتوضيح الخطوات (بنفس لغة السؤال).
2. إذا كانت المسألة تتعلق برسم (مثل دائرة، قطع مكافئ، حركة جسم، مجال كهربائي...)، ولّد كود Python باستخدام matplotlib أو sympy لرسم الشكل المطلوب.
3. اجعل الكود يحفظ الرسم كصورة باسم "plot.png" بدلاً من plt.show().

السؤال:
{data.input_Q}

الرجاء إرجاع الرد بهذا الشكل:

<شرح_الحل>

```python
<كود_Python_لرسم_الشكل>
"""
    response = gemini_model.invoke(prompt)
    full_output = getattr(response, 'content', "").strip()

    try:
        explanation, code = full_output.strip().split("```python")
        code = code.split("```")[0]
    except:
        explanation = full_output
        code = "# لا يوجد كود رسم مناسب."

    script_path = "generated_plot.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)

    try:
        subprocess.run(["python", script_path], check=True)
        image_generated = os.path.exists("plot.png")
    except subprocess.CalledProcessError:
        image_generated = False

    return {
        "solution": explanation.strip(),
        "drawing_code": code.strip(),
        "python_file": script_path,
        "image_generated": image_generated
    }

from fastapi.responses import FileResponse

@app.get("/quiz/download/plot.py")
def download_python_file():
    return FileResponse("generated_plot.py", media_type="text/x-python", filename="plot.py")




#==================================================================================================================================================================
#==================================================================================================================================================================
#==================================================================================================================================================================
#==================================================================================================================================================================



# Request Models
class QuestionRequest(BaseModel):
    question: str
    prev_question: str = ""
    use_existing_index: bool = True 

class UploadResponse(BaseModel):
    message: str
    index_path: str

# Initialize embeddings
hf_embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Global vectorstore instance
vectorstore = None

def clean_response(text):
    return re.sub(r"[ﭼ-ﯿ]+", "", text)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def initialize_vectorstore(pdf_path: Optional[str] = None, index_path: str = DEFAULT_FAISS_INDEX):
    global vectorstore
    
    if os.path.exists(index_path) and pdf_path is None:
        print("Loading existing FAISS index...")
        vectorstore = FAISS.load_local(index_path, hf_embeddings, allow_dangerous_deserialization=True)
    elif pdf_path:
        print("Processing new PDF...")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
        splits = text_splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(splits, hf_embeddings)
        vectorstore.save_local(index_path)
        print(f"Saved new index to {index_path}")
    else:
        raise HTTPException(status_code=400, detail="Neither PDF nor existing index provided")
    
    return vectorstore

#LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

prompt = ChatPromptTemplate.from_template("""
أنت مساعد ذكي يساعد الطلاب في جميع المواد الدراسية. إذا كان السؤال يحتوي على معادلات أو مسائل رياضية أو علمية، قم بحلها بدقة حتى لو لم تكن موجودة في النص. 
ركّز على الفهم والتحليل، ولا تعتمد فقط على السياق النصي.

السياق:
{context}

السؤال السابق: {prev_question}
السؤال الحالي: {question}

الإجابة:
""")


#endpoints
@app.post("/chatpdf/upload_pdf")
async def upload_pdf(file: UploadFile = File(...), index_path: str = DEFAULT_FAISS_INDEX):
    try:
        # Save uploaded PDF 
        temp_pdf = "temp_uploaded.pdf"
        with open(temp_pdf, "wb") as f:
            f.write(await file.read())
        
        # Process PDF 
        initialize_vectorstore(pdf_path=temp_pdf, index_path=index_path)
        os.remove(temp_pdf)  
        
        return UploadResponse(
            message="تم معالجة الملف وحفظ الفهرس بنجاح",
            index_path=index_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chatpdf/ask")
async def ask_question(request: QuestionRequest, index_path: str = DEFAULT_FAISS_INDEX):
    global vectorstore
    
    try:
        if vectorstore is None:
            initialize_vectorstore(index_path=index_path)
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 40})
        
        chain = (
            {
                "context": retriever | format_docs,
                "prev_question": lambda x: request.prev_question,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        response = chain.invoke(request.question)
        return {"answer": clean_response(response)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chatpdf/status")
async def check_status(index_path: str = DEFAULT_FAISS_INDEX):
    index_exists = os.path.exists(index_path)
    return {
        "index_exists": index_exists,
        "index_path": index_path,
        "ready_for_queries": index_exists or (vectorstore is not None)
    }



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # default only for local
    uvicorn.run("quiz:app", host="0.0.0.0", port=port)
