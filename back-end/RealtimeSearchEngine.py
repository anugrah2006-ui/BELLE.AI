from googlesearch import search
from groq import Groq
from json import load, dump
import datetime
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
    raise RuntimeError(f"groq_api_key missing in {env_path}")

client = Groq(api_key=groq_api_key)

# ================== SYSTEM PROMPT ==================
SYSTEM_PROMPT = f"""
Hello, I am {username}.
You are an advanced AI assistant named {assistant_name}.
Respond professionally with clear formatting and accuracy.
"""

# ================== CHAT MEMORY ==================
chatlog_path = os.path.join(script_dir, "..", "Data", "ChatLog.json")

try:
    with open(chatlog_path, "r") as f:
        messages = load(f)
except (FileNotFoundError, ValueError):
    messages = []
    with open(chatlog_path, "w") as f:
        dump(messages, f)

# ================== HELPERS ==================
def google_search_results(query):
    try:
        # googlesearch-python returns a generator of URLs (strings)
        results = list(search(query, num_results=5))
        if not results:
            return "No Google search results found."
            
        text = f"Search results for: {query}\n\n"
        for i, r in enumerate(results, 1):
            if isinstance(r, str):
                # If it's a string, it's a URL
                text += f"Result {i}: {r}\n"
            else:
                # If it's an object with attributes
                title = getattr(r, 'title', 'Search Result')
                description = getattr(r, 'description', 'No description available')
                url = getattr(r, 'url', str(r))
                text += f"Title: {title}\nDescription: {description}\nURL: {url}\n\n"
        return text
    except Exception as e:
        return f"Google Search Error: {str(e)}"

def clean_answer(answer):
    return "\n".join(line for line in answer.split("\n") if line.strip())

def realtime_info():
    now = datetime.datetime.now()
    return (
        "Real-time information:\n"
        f"Day: {now.strftime('%A')}\n"
        f"Date: {now.strftime('%d')}\n"
        f"Month: {now.strftime('%B')}\n"
        f"Year: {now.strftime('%Y')}\n"
        f"Time: {now.strftime('%H:%M:%S')}\n"
    )

# ================== CORE ENGINE ==================
def RealtimeSearchEngine(prompt):
    global messages

    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *messages,
        {
            "role": "system",
            "content": google_search_results(prompt) + realtime_info()
        },
        {"role": "user", "content": prompt}
    ]

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=conversation,
        temperature=0.7,
        max_tokens=1000,
        top_p=0.9,
        stream=True
    )

    answer = ""
    for chunk in completion:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            answer += delta.content

    answer = answer.strip()

    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": answer})

    with open(chatlog_path, "w") as f:
        dump(messages, f, indent=4)

    return clean_answer(answer)

# ================== RUN ==================
if __name__ == "__main__":
    while True:
        prompt = input("Enter your query: ")
        if prompt.lower() in ("exit", "quit"):
            break
        print("\n" + RealtimeSearchEngine(prompt) + "\n")
