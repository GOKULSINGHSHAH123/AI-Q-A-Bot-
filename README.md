# 🤖 AI Q&A Bot – Powered by Google Gemini & LangChain

An interactive **Streamlit-based chatbot** that uses **Google Gemini 2.0 Flash** via **LangChain** to provide intelligent, context-aware, and streaming conversational responses.

---

## 📁 Project Structure

```
.
├── app.py                 # Main Streamlit UI and application logic
├── chatbot_backened.py    # Chatbot backend logic (LangChain + Gemini)
├── config.py              # Configuration file for model and app settings
└── README.md              # Documentation (this file)
```

---

## 🚀 Features

* **Google Gemini 2.0 Flash Integration**
  Uses the latest Gemini model for conversational AI.

* **LangChain Framework**
  Handles memory, prompts, and streaming responses efficiently.

* **Streamlit Interface**
  A simple, responsive UI for real-time chat.

* **Memory Management**
  Retains past conversation context for continuity.

* **Streaming Responses**
  Displays token-by-token generation for a live chat feel.

* **Customizable Parameters**
  Adjust temperature, max tokens, and reset chat memory.

---

## 🧠 How It Works

The project integrates **LangChain’s ConversationChain** with Google’s **Gemini model**, all wrapped in a **Streamlit web app**.

### Step 1: Configuration (`config.py`)

* Defines the `AppConfig` dataclass with all app parameters:

  * Model name (`MODEL_NAME`)
  * Temperature (creativity control)
  * Token limit (`MAX_TOKENS`)
  * UI layout settings

* Provides a method `get_api_key()` to fetch your **Google API key** from:

  * Environment variables (`GOOGLE_API_KEY`)
  * Streamlit secrets

```python
api_key = config.get_api_key()
```

---

### Step 2: Chatbot Backend (`chatbot_backened.py`)

This file handles all **LangChain logic and chatbot behavior**.

#### Key Components:

1. **`StreamlitCallbackHandler`**

   * Streams tokens live to the frontend.
   * Displays partial text updates (`▌` cursor effect).

2. **`GeminiChatBot`**

   * Initializes the Gemini model via `ChatGoogleGenerativeAI`.
   * Uses `ConversationBufferMemory` for context persistence.
   * Custom `PromptTemplate` defines dialogue style and context.
   * Supports streaming and non-streaming responses.

3. **Methods Overview**

   * `get_response()` → Generates model output.
   * `clear_memory()` → Clears all past chat history.
   * `get_conversation_history()` → Retrieves stored conversation context.
   * `add_system_context()` → Injects custom behavior context (e.g., “You are a helpful assistant”).

4. **`initialize_chatbot()`**

   * Handles safe initialization with proper error messages.
   * Adds predefined system instructions.

---

### Step 3: Streamlit Application (`app.py`)

This is the **entry point** that runs the chatbot web interface.

#### Main Components:

1. **App Configuration**

   * Configures Streamlit page settings (`st.set_page_config`).
   * Loads `AppConfig` from `config.py`.

2. **UI Styling**

   * `render_custom_css()` adds modern, responsive styling.
   * Custom chat message cards for user and assistant.

3. **Sidebar Controls**

   * Input Google API key.
   * Adjust temperature and token limits.
   * Buttons to clear or reset chat memory.
   * Tips for best use.

4. **Chat Interface**

   * Displays past messages from `st.session_state`.
   * Collects user prompts using `st.chat_input`.
   * Uses `GeminiChatBot.get_response()` for real-time streaming output.
   * Shows metrics like model, temperature, and message count.

5. **Footer Section**

   * Includes credits and timestamp.

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-qa-bot.git
cd ai-qa-bot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

*(If you don’t have one, create `requirements.txt` with the following:)*

```
streamlit
langchain
langchain-google-genai
google-generativeai
dataclasses
```

### 4. Set Up API Key

Get your **Google Gemini API key** from [Google AI Studio](https://aistudio.google.com/app/apikey).

You can set it in your environment:

```bash
export GOOGLE_API_KEY="your_api_key_here"
```

or inside Streamlit secrets (`.streamlit/secrets.toml`):

```toml
GOOGLE_API_KEY = "your_api_key_here"
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Then open the app in your browser (default: [http://localhost:8501](http://localhost:8501)).

---

## 💬 Example Questions

| Type      | Example                                |
| --------- | -------------------------------------- |
| Technical | "Explain difference between AI and ML" |
| Creative  | "Write a story about an AI assistant"  |
| Code Help | "Help me debug this Python function"   |

---

## 🧹 Common Actions

| Action             | Description                                      |
| ------------------ | ------------------------------------------------ |
| 🧹 Clear Chat      | Clears memory and message history                |
| 🔄 Reset Bot       | Re-initializes chatbot                           |
| ⚙️ Adjust Settings | Modify temperature and token length from sidebar |

---

## 🧩 File-by-File Summary

| File                  | Purpose                                                          |
| --------------------- | ---------------------------------------------------------------- |
| `app.py`              | Streamlit UI, chatbot orchestration, and state management        |
| `chatbot_backened.py` | Backend chatbot logic (LLM, memory, prompt, response generation) |
| `config.py`           | Central configuration and environment key management             |

---

## 🛠️ Troubleshooting

| Issue                       | Possible Fix                                                             |
| --------------------------- | ------------------------------------------------------------------------ |
| `No API Key Found`          | Set the `GOOGLE_API_KEY` in `.env` or Streamlit secrets                  |
| `Error generating response` | Ensure the Gemini API key is valid and the internet connection is active |
| `App not loading`           | Verify Streamlit is installed and run `streamlit run app.py` again       |

---

## 🧾 License

This project is for **educational and internal use**. Modify and adapt freely for learning or internal demos.

---

## ❤️ Acknowledgments

* [LangChain](https://www.langchain.com/)
* [Google Gemini](https://aistudio.google.com/)
* [Streamlit](https://streamlit.io/)

---
