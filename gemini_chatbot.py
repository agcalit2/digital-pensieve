from google import genai
import os
from typing import Optional
from dotenv import load_dotenv

# Configure the Gemini API
def configure_gemini(api_key: Optional[str] = None):
    """Configure the Gemini API with the provided API key."""
    # Load environment variables from .env file
    load_dotenv()
    
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Please set it in the .env file or pass it as an argument\n"
                "Create a .env file in the project root with: GEMINI_API_KEY=your-api-key"
            )
    return genai.Client(api_key=api_key)

def get_gemini_response(prompt: str, model_name: str = "gemini-1.5-pro") -> str:
    """Get a response from the Gemini model."""
    model = (model_name)
    response = model.generate_content(prompt)
    return response.text

def chat_loop(client):
    """Run the chat loop."""
    print("Welcome to Gemini Chat! Type 'quit' or 'exit' to end the session.")
    print("You're chatting with Gemini. Start typing your messages...\n")
    
    while True:
        try:
            # Get user input
            user_input = input("You: ")
            
            # Check for exit conditions
            if user_input.lower() in ('quit', 'exit'):
                print("Goodbye!")
                break
                
            if not user_input.strip():
                continue
                
            # Get and display response
            print("\nGemini: ", end="", flush=True)
            response = client.models.generate_content(
                model="gemini-2.0-flash", contents=user_input
            )
            print(f"{response.text}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}\n")

def main():
    try:
        client = configure_gemini()
        print("Successfully connected to Gemini API!")
        chat_loop(client)
    except ValueError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
