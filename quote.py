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

def quote(description: str, interests: list[str]) -> QuoteResponse:
    interests_str = ", ".join(interests)
    user_message = f"Topic: {description}\nInterests: {interests_str}"
    
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a quote generator. You will generate a list of quotes with the author, location of the author and an interpretation of the quote for the user. You will be given a description of the topic, a style of the quote, a list of interests and the language to generate them in. Return exactly 5 quotes.",
            },
            {"role": "user", "content": user_message},
        ],
        response_format=QuoteResponse,
    )
    
    return completion.choices[0].message.parsed