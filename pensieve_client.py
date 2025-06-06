from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="python",  # Executable
    args=["main.py"],  # Optional command line arguments
    env=None,  # Optional environment variables
)

async def run():
  async with stdio_client(server_params) as (read, write):
    async with ClientSession(
        read, write
    ) as session:
      # Initialize the connection
      await session.initialize()

      # List available prompts
      prompts = await session.list_prompts()

      # List available resources
      resources = await session.list_resources()

      # List available tools
      tools = await session.list_tools()

      print("Available prompts:", prompts)
      print("Available resources:", resources)
      print("Available tools:", tools)

      print("Attempting to Call Resource")
      print(await session.read_resource("memory:://"))

def main():
    import asyncio
    import sys
    
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()