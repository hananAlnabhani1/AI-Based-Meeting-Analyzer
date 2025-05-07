# 🎙️ Smart Meeting Assistant – AI-Powered Meeting Analyzer

Smart Meeting Assistant is an end-to-end AI system that **transcribes**, **summarizes**, and **answers questions** about your meeting recordings. Upload any audio file and interact through a chat interface powered by RAG (Retrieval-Augmented Generation) and LangSmith tracing.

---

## 🚀 Features

- 🎧 **Universal Audio-to-Text**  
  Upload MP3/WAV and get a complete transcript via Whisper.
- 🧠 **Long-Form Summarization**  
  Automatically chunk & summarize multi-hour meetings into bite-sized overviews.
- 📋 **Structured Minutes Extraction**  
  Extract decisions, owners, deadlines and notes as Markdown bullet lists.
- 🤖 **Retrieval-Augmented Chat**  
  Ask follow-up questions about any part of the meeting; powered by ChromaDB + OpenAI.
- 🔍 **Meeting Classification & Date Extraction**  
  Auto-label your meeting type (e.g., “Finance Mtg”) and pull out dates mentioned.
- 📊 **LangSmith Tracing & Evaluation**  
  All LLM calls are traced in LangSmith for observability and testing.

---

## 🧰 Technologies Used

- **Gradio** for quick, interactive UI  
- **WhisperModel** (faster-whisper) for speech-to-text  
- **Hugging Face Transformers** (distilbart-cnn) for summarization  
- **LangChain** (chains, retriever, chat, text splitter)  
- **ChromaDB** for vector embeddings & retrieval  
- **OpenAI GPT** via LangChainOpenAI for classification, structured prompts, and chat  
- **LangSmith** for run tracing, dataset & evaluation integration  
- **Pydub** for audio chunking  
- **Python-Dotenv** for secret management  

---

## 📂 How It Works

1. **Input**  
   - User uploads an audio file (MP3/WAV).  
2. **Transcription**  
   - Audio is split into 2-minute chunks and each chunk is transcribed by Whisper.  
3. **Summarization & Structuring**  
   - The full transcript is chunked, summarized, and run through structured prompts to extract bullets.  
4. **Vector Database**  
   - Transcript is loaded into ChromaDB as embeddings for retrieval.  
5. **RAG Agent**  
   - A RetrievalQA chain answers chat queries by fetching top-k relevant chunks and using GPT.  
6. **Tracing & Evaluation**  
   - Every LLM call is logged in LangSmith for performance metrics, debugging, and drift detection.

---

## 🧠 Memory & Agent Workflow

- **Stateful Chat** via `gr.State()` holds the RetrievalQA agent between chat turns.  
- On each user question:
  1. Query the vector retriever for relevant transcript snippets.  
  2. Invoke GPT to produce an answer using those snippets as context.  
  3. Append both question & answer to the chat history.  

---

## 🖼️ Example Use Cases

| Action                                    | Example                                                                 |
|-------------------------------------------|-------------------------------------------------------------------------|
| **Transcribe & Summarize**                | Upload "team_meeting.mp3" → get transcript + short & structured summary. |
| **Ask About a Decision**                  | “What did we decide about the budget?” → GPT answers with sources.      |
| **Find Responsible Person**               | “Who owns the security policy update?” → GPT cites name & task.        |
| **Retrieve Deadline**                     | “When is the next deliverable due?” → GPT pulls date from transcript.   |
| **Classify Meeting**                      | “What type of meeting was this?” → “Engineering Stand-up”               |

---

## 📦 Setup Instructions

```bash
git clone https://github.com/YOUR_USERNAME/SmartMeetingAssistant.git
cd SmartMeetingAssistant

# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure secrets
echo "OPENAI_API_KEY=sk-..." > .env
echo "LANGCHAIN_TRACING_V2=true" >> .env
echo "LANGCHAIN_API_KEY=lsv2-..." >> .env

bash

git clone https://github.com/hananAlnabhani1/AI-Based-Meeting-Analyzer.git
# 3. Run the app
python app.py
