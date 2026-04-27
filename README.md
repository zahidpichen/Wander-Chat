# Wanderchat

Wanderchat is a premium, AI-powered intelligent travel assistant designed to help users explore and plan trips, currently optimized for Kerala tourism. It seamlessly transitions between a "Discovery" phase (suggesting destinations based on interests) and a "Planning" phase (generating detailed itineraries, budgets, and local recommendations).

## 🚀 Tech Stack

### Frontend
- **Framework**: Next.js (React)
- **Styling**: Tailwind CSS (v4) with custom mesh gradients and glassmorphism.
- **Animations**: Framer Motion
- **Icons**: Lucide React

### Backend
- **Framework**: FastAPI (Python)
- **AI Model**: Google Gemini 2.0 Flash (via OpenRouter)
- **Web Search**: Tavily API (for live events, festivals, and local shops)
- **RAG / Vector Database**: FAISS (CPU) with `sentence-transformers`
- **Data Processing**: `pdfplumber` and `python-docx` for local knowledge base ingestion.

## 🛠️ Setup Instructions

1. **Clone the repository**:
   \`\`\`bash
   git clone https://github.com/zahidpichen/Wander-Chat.git
   cd Wander-Chat
   \`\`\`

2. **Environment Variables**:
   Copy the example environment file and add your API keys:
   \`\`\`bash
   cp .env.example .env
   \`\`\`
   Add your OpenRouter and Tavily API keys to the `.env` file.

3. **Backend Setup**:
   \`\`\`bash
   # Create and activate a virtual environment
   python -m venv .venv
   source .venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Run the FastAPI server
   cd backend
   uvicorn main:app --reload --port 8000
   \`\`\`

4. **Frontend Setup**:
   \`\`\`bash
   # Open a new terminal
   cd frontend
   
   # Install dependencies
   npm install
   
   # Start the Next.js development server
   npm run dev
   \`\`\`
   The application will be available at `http://localhost:3000`.

## 🧠 How the AI Works
Wanderchat operates natively using pure Python and FastAPI, without relying on heavy agentic frameworks like LangChain. This ensures maximum performance and complete control over the RAG pipeline. It maintains a short-term memory of the conversation history to understand follow-up queries seamlessly.

---

## 👥 Contributors

This project is built and maintained by our team. If you are a team member, please add your name and USN number below:

- Zahid Pichen - 23BTRCL006
- *[Team Member Name]* - *[USN]*
- *[Team Member Name]* - *[USN]*
- *[Team Member Name]* - *[USN]*
- *[Team Member Name]* - *[USN]*
- *[Team Member Name]* - *[USN]*
