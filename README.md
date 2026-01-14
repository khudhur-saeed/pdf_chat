# PDF Chat Application

A modern web application that allows users to upload PDF documents and have intelligent conversations about their content using AI-powered PDF analysis and multi-language support.

## 🚀 Features

- **PDF Upload & Management**
  - Upload multiple PDF files
  - View all uploaded documents in a sidebar
  - Select which PDF to chat about
  - Delete PDF files from storage
  - Auto-select newly uploaded PDFs

- **Intelligent Chat Interface**
  - Ask questions about PDF content
  - Context-aware responses based on document
  - Multi-session conversation history
  - Real-time typing indicators
  - Beautiful dark-themed UI

- **Message Formatting**
  - Bold text rendering
  - Numbered and bullet lists
  - Proper line breaks and spacing
  - RTL (Right-to-Left) text support for Arabic and other languages

- **Multi-Language Support**
  - Arabic (العربية)
  - English
  - Turkish
  - Extensible language support

- **Session Management**
  - Create new chat sessions
  - Save conversation history
  - Clear all chats
  - Persistent message tracking

## 📋 Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **LangChain** - LLM integration and document processing
- **Google Generative AI** - Gemini API for intelligent responses
- **Chroma** - Vector database for document embeddings
- **Unstructured** - PDF parsing and text extraction

### Frontend
- **React 18** - UI framework
- **TypeScript** - Type-safe JavaScript
- **Vite** - Fast build tool
- **Tailwind CSS** - Utility-first styling
- **shadcn/ui** - High-quality React components
- **Lucide React** - Beautiful icons

## 🛠️ Requirements

### System Requirements
- Python 3.10+
- Node.js 18+
- 2GB+ RAM
- Internet connection (for API calls)

### API Keys
- **Google Gemini API Key** - Get from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/khudhur-saeed/pdf_chat.git
cd pdf_chat
```

### 2. Backend Setup
```bash
# Create and activate virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "GEMINI_KEY=your_api_key_here" > .env
```

### 3. Frontend Setup
```bash
cd Clinet

# Install dependencies
npm install

# Return to root
cd ..
```

## 🚀 Running the Application

### Start Backend
```bash
source env/bin/activate  # On Windows: env\Scripts\activate
python main.py
```
Backend runs on: `http://localhost:8000`

### Start Frontend (in a new terminal)
```bash
cd Clinet
npm run dev
```
Frontend runs on: `http://localhost:3000`

### Access the Application
Open your browser and go to: **http://localhost:3000**

## 📖 Usage Guide

### 1. Upload a PDF
- Click the **"📤 Upload PDF"** button in the top-right corner
- Select a PDF file from your computer
- The file will be processed and added to the "Uploaded PDFs" list

### 2. Select a PDF to Chat About
- In the left sidebar under "Uploaded PDFs", click on any PDF file
- A checkmark (✓) will appear next to the selected PDF
- The header will show which PDF you're currently chatting about

### 3. Start a Conversation
- Click **"New chat"** to create a new conversation
- Type your question about the PDF in the input box
- The PDF Assistant will analyze the document and provide answers
- Messages are formatted with bold text, lists, and proper spacing

### 4. Manage Files
- **Delete a PDF**: Hover over a PDF in the sidebar and click the trash icon
- **View Chat History**: All conversations are saved in the sidebar
- **Clear All Chats**: Click the trash icon in the header to clear history

## 🔌 API Endpoints

### Chat & Sessions
- `POST /create-session` - Create a new chat session
- `POST /chat` - Send a message and get AI response
- `GET /session/{session_id}/history` - Get conversation history
- `DELETE /session/{session_id}` - Clear a session

### PDF Management
- `POST /upload-pdf` - Upload a new PDF file
- `POST /select-pdf/{filename}` - Select a PDF to chat about
- `GET /uploaded-files` - List all uploaded PDFs
- `DELETE /uploaded-files/{filename}` - Delete a PDF file
- `GET /current-pdf` - Get currently selected PDF
- `GET /pdf-status` - Check if a PDF is loaded

### System
- `GET /stats` - Get system statistics
- `GET /docs` - Swagger API documentation

## 📁 Project Structure

```
pdf_chat/
├── main.py                 # FastAPI backend
├── .env                    # Environment variables (create this)
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── Clinet/                # React frontend
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── styles/        # Global styles
│   │   └── main.tsx       # Entry point
│   └── package.json       # npm dependencies
├── uploads/               # Uploaded PDF storage
├── vector_db/            # Vector database (Chroma)
└── env/                  # Python virtual environment
```

## 🔑 Environment Variables

Create a `.env` file in the root directory:

```env
GEMINI_KEY=your_google_gemini_api_key_here
```

## ⚙️ Configuration

### Model Selection
Edit `main.py` to change the AI model:
```python
self.llm = GoogleGenerativeAI(
    model="gemini-2.0-flash",  # Change this
    api_key=api_key
)
```

Available models:
- `gemini-2.0-flash` (faster, default)
- `gemini-1.5-pro` (more powerful)
- Other Google Gemini models

### Embedding Model
```python
self.embedding = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004"  # For document embeddings
)
```

## 🚨 Troubleshooting

### API Quota Exceeded
**Error:** "429 RESOURCE_EXHAUSTED"
- **Solution:** Wait a moment and try again, or get a new API key with higher limits

### Port Already in Use
**Error:** "Address already in use"
- **Solution:** Kill the process or use a different port
```bash
lsof -ti:8000 | xargs kill -9  # Kill backend
lsof -ti:3000 | xargs kill -9  # Kill frontend
```

### PDF Not Loading
- Ensure the PDF is valid and not corrupted
- Check that the file size is reasonable (< 100MB recommended)
- Verify the `uploads/` directory has read/write permissions

### CORS Errors
- Backend CORS is configured for all origins (`*`)
- If issues persist, check FastAPI CORS middleware settings in `main.py`

## 📊 Performance Tips

1. **Large PDFs**: For PDFs > 50MB, consider splitting them
2. **Rate Limiting**: Free tier has rate limits; consider upgrading Gemini API
3. **Vector Database**: First query on a new PDF takes longer (building embeddings)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Screenshots
<img width="1920" height="1080" alt="Screenshot_20260114_152115" src="https://github.com/user-attachments/assets/ddf08e5c-50fe-484b-b6fb-c7a9bd65f50b" />


## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 👤 Author

**Khudhur Saeed**
- GitHub: [@khudhur-saeed](https://github.com/khudhur-saeed)
