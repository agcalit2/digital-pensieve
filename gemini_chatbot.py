from google import genai
from google.genai import types
import os
from typing import Optional
from dotenv import load_dotenv
from google.genai.types import FunctionDeclaration, Schema, JSONSchema
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="python",  # Executable
    args=["main.py"],  # Optional command line arguments
    env=None,  # Optional environment variables
)

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

async def chat_loop(client):
    """Run the chat loop."""
    print("\n" + "="*60)
    print("🚀 GEMINI CHAT INTERFACE".center(60))
    print("="*60)
    print("\nWelcome to Gemini Chat! Type 'quit' or 'exit' to end the session.")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read, write
        ) as session:

            # Initialize the connection
            print("🔌 Connecting to MCP server...")
            await session.initialize()

            # List available prompts
            p = await session.list_prompts()

            # List available resources
            r = await session.list_resources()

            # List available tools
            t = await session.list_tools()
            google_tools = []
            for tool in t.tools:
                try:
                    # Convert the JSON schema to a Gemini Schema object
                    schema = Schema.from_json_schema(json_schema=JSONSchema(**tool.inputSchema))
                    
                    # Create the function declaration with the converted schema
                    func_decl = FunctionDeclaration(
                        name=tool.name,
                        description=tool.description or "",
                        parameters=schema
                    )
                    
                    # Add the tool with the function declaration
                    google_tools.append(types.Tool(function_declarations=[func_decl]))
                except Exception as e:
                    print(f"⚠️ Failed to convert schema for tool '{tool.name}': {str(e)}")
                    continue

            print("\n📋 Available Memory Components:")
            print("-"*60)
            print(f"📝 Prompts: {len(p.prompts)} available")
            print(f"💾 Resources: {len(r.resources)} available")
            print(f"🛠️  Tools: {len(t.tools)} available")
            print("-"*60 + "\n")

            print("🔍 Attempting to access memory resource...")
            try:
                memory = await session.read_resource("memory:://")
                print("✅ Successfully retrieved memory resource")
                print(f"   Memories: {memory}")
            except Exception as e:
                print(f"❌ Failed to access memory resource: {str(e)}")

            print("You're now chatting with Gemini. Start typing your messages...\n")
            
            # Initialize conversation history
            conversation_history = []
            
            while True:
                try:
                    # Get user input
                    user_input = input("You: ")

                    # Check for exit conditions
                    if user_input.lower() in ('quit', 'exit'):
                        print("\n👋 Goodbye! Have a great day! 😊")
                        break

                    if not user_input.strip():
                        continue
                        
                    # Add user message to conversation history
                    conversation_history.append(types.Content(
                        role="user",
                        parts=[types.Part(**{"text": user_input})]
                    ))

                    # Process the conversation with the model
                    await process_conversation(client, conversation_history, google_tools, session)

                except KeyboardInterrupt:
                    print("\n👋 Goodbye! Have a great day! 😊")
                    break
                except Exception as e:
                    print(f"\n⚠️  An error occurred: {str(e)}\n")

SYSTEM_PROMPT = """
You are a memory storage machine. Your job is to create, read, update, and delete memories based on the desires of the user. Avoid tampering with memories without explicit user intent. When retrieving the memories, try to summarize their contents in a paragraph or two.
"""
async def process_conversation(client, conversation_history, google_tools, session):
    """Handle the conversation with the model, including function calling."""
    print("\n\033[1;35mGemini:\033[0m ", end="", flush=True)
    max_iterations = 5  # Prevent infinite loops

    for _ in range(max_iterations):
        try:
            # Generate the response with function calling enabled
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20",
                contents=conversation_history,
                config=types.GenerateContentConfig(
                    tools=google_tools,
                    system_instruction=SYSTEM_PROMPT
                )
            )

            # Process the response
            if not response.candidates:
                print("\n⚠️ No response generated")
                return

            candidate = response.candidates[0]
            if not candidate.content or not candidate.content.parts:
                print("\n⚠️ Empty response from model")
                return

            function_called = False
            for part in candidate.content.parts:
                if part.function_call:
                    function_called = True
                    await handle_function_call(candidate.content, part, conversation_history, session)
                elif part.text:
                    display_text_response(part.text, conversation_history)

            if not function_called:
                return
                
        except Exception as e:
            print(f"\n⚠️  Error generating response: {str(e)}")
            return

async def handle_function_call(content, part, conversation_history, session):
    """Handle a function call from the model."""
    func_name = part.function_call.name
    func_args = part.function_call.args

    print(f"\n🔧 Calling function: {func_name}")
    print(f"   Arguments: {func_args}")

    conversation_history.append(content)

    try:
        tool_response = await session.call_tool(
            name=func_name,
            arguments=func_args
        )
        response_part = types.Part.from_function_response(
            name=func_name,
            response={"result": tool_response}
        )
        print("✅ Function call successful")
        print(f"Tool Response: {tool_response}")
    except Exception as e:
        error_msg = f"Error calling function {func_name}: {str(e)}"
        print(f"\n⚠️ {error_msg}")
        response_part = types.Part.from_function_response(
            name=func_name,
            response={"error": error_msg}
        )

    # Add the function call and response to the conversation history
    conversation_history.append(
        types.Content(
            role="user",
            parts=[response_part]
        )
    )

def display_text_response(text, conversation_history):
    """Display a text response and add it to the conversation history."""
    conversation_history.append(
        types.Content(
            role="model",
            parts=[types.Part(text=text)]
        )
    )
    formatted_response = text.replace('\n', '\n    ')
    print(f"\n    {formatted_response}\n")
    print("─" * 60)  # Separator line

def main():
    import asyncio
    try:
        client = configure_gemini()
        print("✅ Successfully connected to Gemini API!")
        print("\n💡 Starting chat session...")
        print("   Type your message and press Enter to chat")
        print("   Use 'quit' or 'exit' to end the session\n")
        loop = asyncio.get_event_loop()
        loop.run_until_complete(chat_loop(client))
    except ValueError as e:
        print(f"\n❌ Error: {str(e)}")
    except Exception as e:
        print(f"\n⚠️ An unexpected error occurred: {str(e)}")
    finally:
        print("\n✨ Thank you for using Gemini Chat! ✨\n")

if __name__ == "__main__":
    main()
