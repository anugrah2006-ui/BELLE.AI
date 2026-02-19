# ==========================================
# BELLE.AI Trend Engine
# Social Trend + Grooming Intelligence Layer
# ==========================================

import requests
from dotenv import dotenv_values
from groq import Groq

# ==============================
# LOAD ENVIRONMENT VARIABLES
# ==============================
env = dotenv_values(".env")

GROQ_API_KEY = env.get("groq_api_key")
SERPAPI_API_KEY = env.get("SERPAPI_API_KEY")
APIFY_API_KEY = env.get("APIFY_API_KEY")
HF_API_KEY = env.get("Hugging_Face_api")

client = Groq(api_key=GROQ_API_KEY)

# ==============================
# SYSTEM PROMPT FOR BELLE
# ==============================
BELLE_TREND_PROMPT = """
You are BELLE.AI Trend Mentor.

Your job:
- Help introverted or under-confident people improve grooming and style.
- Be supportive, motivating and emotionally intelligent.
- NEVER shame appearance.
- Focus on growth, comfort, and confidence.
"""

# ==============================
# SERPAPI GOOGLE TRENDS FETCH
# ==============================
def fetch_google_trends(query: str):

    if not SERPAPI_API_KEY:
        return "No Google trends data available."

    try:
        url = "https://serpapi.com/search.json"

        params = {
            "engine": "google_trends",
            "q": query,
            "api_key": SERPAPI_API_KEY
        }

        response = requests.get(url, params=params, timeout=20)
        data = response.json()

        return str(data)[:800]

    except Exception as e:
        return f"Google Trends Error: {str(e)}"


# ==============================
# APIFY SOCIAL TREND PLACEHOLDER
# ==============================
def fetch_social_trends(query: str):

    if not APIFY_API_KEY:
        return "No social media trend data available."

    # Later replace with real APIFY actor call
    return f"Instagram, Pinterest and YouTube trends related to '{query}'."


# ==============================
# HUGGING FACE SUMMARIZER
# ==============================
def hf_summarize(text):

    if not HF_API_KEY:
        return text

    try:
        API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

        headers = {
            "Authorization": f"Bearer {HF_API_KEY}"
        }

        payload = {
            "inputs": f"Summarize grooming trends:\n{text}"
        }

        res = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        data = res.json()

        if isinstance(data, list):
            return data[0]["summary_text"]

        return text

    except Exception:
        return text


# ==============================
# MAIN TREND ENGINE FUNCTION
# ==============================
def trend_engine(user_query: str):

    # Step 1 — Fetch data
    google_trends = fetch_google_trends(user_query)
    social_trends = fetch_social_trends(user_query)

    combined_data = f"""
User Topic: {user_query}

Google Trends:
{google_trends}

Social Media Trends:
{social_trends}
"""

    # Step 2 — HuggingFace Summarization Layer
    summarized_data = hf_summarize(combined_data)

    # Step 3 — BELLE AI Response Generation (Groq)
    messages = [
        {"role": "system", "content": BELLE_TREND_PROMPT},
        {"role": "user", "content": summarized_data}
    ]

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.7,
        max_tokens=700,
        stream=False
    )

    return completion.choices[0].message.content


# ==============================
# TEST MODE
# ==============================
if __name__ == "__main__":
    print("\n--- BELLE Trend Engine Test ---\n")

    query = input("Enter trend topic: ")
    result = trend_engine(query)

    print("\nBELLE says:\n")
    print(result)
