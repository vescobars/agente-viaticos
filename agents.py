import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

MODEL = os.environ.get("GEMINI_MODEL")
API_KEY = os.environ.get("GEMINI_API_KEY")
INSTRUCTIONS = (
    "Eres un asistente amable, cálido y paciente. "
    "Responde siempre en español, en un máximo de 3 oraciones. "
    "Si no sabes la respuesta, dilo con honestidad en lugar de inventar."
)

if not API_KEY or not MODEL:
    raise SystemExit("Missing GEMINI_API_KEY or GEMINI_MODEL. Add them to your .env file.")

client = genai.Client(api_key=API_KEY)


def ask(question: str) -> str:
    interaction = client.interactions.create(
        model=MODEL,
        input=question,
        system_instruction=INSTRUCTIONS,
    )
    return interaction.output_text


if __name__ == "__main__":
    question = input("Pregunta: ")
    print(ask(question))
