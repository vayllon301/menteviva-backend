from openai import OpenAI
from dotenv import load_dotenv
import os
import time

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Your assistant ID
ASSISTANT_ID = "asst_Iz3pZAIB1OOVeyyKfx3QTrmS"

def blanca(message: str):
    """
    Send a message to the OpenAI assistant and get a response.
    
    Args:
        message: The user's message
        
    Returns:
        The assistant's response
    """
    try:
        # Create a thread for the conversation
        thread = client.beta.threads.create()
        
        # Add the message to the thread
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=message
        )
        
        # Run the assistant
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=ASSISTANT_ID
        )
        
        # Wait for the run to complete
        while run.status != "completed":
            time.sleep(0.5)
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
        
        # Get the messages from the thread
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        
        # Return the last assistant message
        for msg in messages.data:
            if msg.role == "assistant":
                return msg.content[0].text
        
        return "No hubo respuesta del asistente"
        
    except Exception as e:
        return f"Error: {str(e)}"
