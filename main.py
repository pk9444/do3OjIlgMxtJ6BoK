from dotenv import load_dotenv
import os

load_dotenv() # Loads variables from .env by default

api_key = os.getenv("OPENAI_API_KEY")

print(api_key)
