import sys
import os
import re

# Add back-end to path
sys.path.append(os.path.join(os.path.dirname(__file__), "back-end"))

from Model import FirstLayerDMM
from Chatbot import ChatBot
from RealtimeSearchEngine import RealtimeSearchEngine
from trend_engine import trend_engine
from Image_Analyzer import analyze_image
from ImageGenration import GenerateImages
from dotenv import dotenv_values

env_vars = dotenv_values(".env")
assistant_name = env_vars.get("Assistant_name", "BELLE")

print(f"\n===== {assistant_name}.AI Online =====\n")

def clean_query(task, prefix):
    """Cleans the task string by removing the prefix and any parentheses."""
    query = task.replace(prefix, "", 1).strip()
    # Remove parentheses if they wrap the query (e.g., "( query )")
    query = re.sub(r'^\(\s*(.*?)\s*\)$', r'\1', query)
    return query.strip()

while True:
    try:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        # Exit condition
        if user_input.lower() in ["exit", "quit", "bye"]:
            print(f"\n{assistant_name}: Take care. I'm always here for you 🌸")
            break

        # Step 1 — Decision Model (Classify query into tasks)
        tasks = FirstLayerDMM(user_input)

        if not tasks:
            # Fallback to general chat if no task is detected
            print(f"\n{assistant_name}:\n{ChatBot(user_input)}\n")
            continue

        final_response = ""
        already_handled = False

        # Step 2 — Execute tasks
        for task in tasks:
            task = task.lower().strip()

            # ------------------------------
            # EXIT
            # ------------------------------
            if "exit" in task:
                print(f"\n{assistant_name}: Take care. I'm always here for you 🌸")
                sys.exit()

            # ------------------------------
            # GENERAL CHAT
            # ------------------------------
            elif task.startswith("general"):
                query = clean_query(task, "general")
                final_response += ChatBot(query) + "\n"
                already_handled = True

            # ------------------------------
            # REALTIME SEARCH
            # ------------------------------
            elif task.startswith("realtime"):
                query = clean_query(task, "realtime")
                final_response += RealtimeSearchEngine(query) + "\n"
                already_handled = True

            # ------------------------------
            # TREND ENGINE
            # ------------------------------
            elif task.startswith("trend"):
                query = clean_query(task, "trend")
                final_response += trend_engine(query) + "\n"
                already_handled = True

            # ------------------------------
            # IMAGE GENERATION
            # ------------------------------
            elif task.startswith("generate image"):
                query = clean_query(task, "generate image")
                GenerateImages(query)
                final_response += f"I've generated the images for '{query}'. 🎨\n"
                already_handled = True

            # ------------------------------
            # IMAGE ANALYZER
            # ------------------------------
            elif "image" in task:
                try:
                    # For image analysis, we use the cleaned prompt if possible
                    print(f"\n[INFO] Opening image selection dialog...")
                    result = analyze_image(user_input) 
                    final_response += result + "\n"
                    already_handled = True
                except Exception as e:
                    final_response += f"Image analysis failed: {str(e)}\n"

            # ------------------------------
            # AUTOMATION (Placeholders/Fallback)
            # ------------------------------
            elif any(task.startswith(func) for func in ["open", "close", "play", "system", "content", "google search", "youtube search", "reminder"]):
                # Handle these intents if they are detected but not implemented yet
                query = task.strip()
                # For now, we still use ChatBot to respond to these requests
                if not already_handled:
                    final_response += ChatBot(user_input) + "\n"
                    already_handled = True
                else:
                    final_response += f"(Detected intent: {task} - Not fully implemented yet)\n"

        # Step 3 — Final Output
        if final_response:
            print(f"\n{assistant_name}:\n")
            print(final_response.strip())
            print()
        elif not already_handled:
            # Final fallback if nothing was processed
            print(f"\n{assistant_name}:\n{ChatBot(user_input)}\n")

    except KeyboardInterrupt:
        print(f"\n\n{assistant_name}: Goodbye!")
        break
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
