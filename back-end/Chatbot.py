from groq import Groq
from json import load, dump
from dotenv import dotenv_values
import os

# ================== PATH SETUP ==================
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, "..", ".env")

env_vars = dotenv_values(env_path)

username = env_vars.get("username", "User")
assistant_name = env_vars.get("Assistant_name", "HIRA")
groq_api_key = env_vars.get("groq_api_key")

if not groq_api_key:
    raise RuntimeError("groq_api_key missing in .env file")

# ================== INIT CLIENT ==================
client = Groq(api_key=groq_api_key)

# ================== SYSTEM PROMPT ==================
SYSTEM_PROMPT = f"""
Hello, I am {username}.
You are a very accurate and advanced AI chatbot named {assistant_name}.

- Do not tell time unless asked.
- Answer concisely.
- Reply only in English.
- Do not provide notes.
- Never mention training data.
"""

systemChatbot = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

# ================== CHAT MEMORY ==================
DATA_DIR = "Data"
CHATLOG_PATH = os.path.join(DATA_DIR, "ChatLog.json")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

try:
    with open(CHATLOG_PATH, "r") as f:
        messages = load(f)
except (FileNotFoundError, ValueError):
    messages = []
    with open(CHATLOG_PATH, "w") as f:
        dump(messages, f, indent=4)

# ================== HELPERS ==================
def AnswerModifier(answer: str) -> str:
    return "\n".join(line for line in answer.split("\n") if line.strip())

# ================== CORE CHATBOT ==================
def ChatBot(query: str) -> str:
    global messages

    try:
        messages.append({"role": "user", "content": query})

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=systemChatbot + messages,
            temperature=0.7,
            max_tokens=1000,
            stream=True
        )

        answer = ""
        for chunk in completion:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                answer += delta.content

        answer = answer.replace("</s>", "").strip()

        messages.append({"role": "assistant", "content": answer})

        with open(CHATLOG_PATH, "w") as f:
            dump(messages, f, indent=4)

        return AnswerModifier(answer)

    except Exception as e:
        print(f"[ERROR] {e}")
        return "Something went wrong. Please try again."

# ================== RUN ==================
if __name__ == "__main__":
    while True:
        user_input = input("Enter your question: ")
        if user_input.lower() in ("exit", "quit"):
            break
        print(ChatBot(user_input))
