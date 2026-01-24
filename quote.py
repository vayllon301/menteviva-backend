from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

class Quote(BaseModel):
    quote: str
    author: str
    location: str
    interpretation: str

class QuoteResponse(BaseModel):
    quotes: list[Quote]

def make_conversational(quote_response: QuoteResponse) -> str:
    """
    Transforma las citas en un mensaje natural y conversacional.
    """
    quotes = quote_response.quotes
    
    # Crear un prompt para hacer el mensaje conversacional
    quotes_text = ""
    for i, quote in enumerate(quotes, 1):
        quotes_text += f"{i}. \"{quote.quote}\" - {quote.author} ({quote.location})\n"
        quotes_text += f"   Interpretación: {quote.interpretation}\n\n"
    
    # Usar OpenAI para convertir a formato conversacional
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Eres un asistente amigable y conversacional. Tu trabajo es tomar una lista de citas y sus interpretaciones, y transformarlas en un mensaje natural, cálido y conversacional en español. Mantén las citas originales pero presenta la información de manera más cercana y amigable. Usa un tono empático y motivador."
            },
            {
                "role": "user",
                "content": f"Transforma estas citas en un mensaje conversacional y natural:\n\n{quotes_text}"
            }
        ],
        temperature=0.7,
    )
    
    return completion.choices[0].message.content

def quote(description: str, interests: list[str]) -> QuoteResponse:
    interests_str = ", ".join(interests)
    user_message = f"Topic: {description}\nInterests: {interests_str}"
    
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a quote generator. You will generate a list of quotes with the author, location of the author and an interpretation of the quote for the user. You will be given a description of the topic, a style of the quote, a list of interests and the language to generate them in. Return exactly 5 quotes in the language provided.",
            },
            {"role": "user", "content": user_message},
        ],
        response_format=QuoteResponse,
    )
    
    return completion.choices[0].message.parsed