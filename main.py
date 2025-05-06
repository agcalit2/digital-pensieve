import time
import os
import threading
import pickle
from mcp.server.fastmcp import FastMCP
from models import similarity_model, qa_model, Memory, Topic

# Create an MCP server
mcp = FastMCP("Pensieve")

# Load existing memories and topics if available
if os.path.exists("pensieve_memories.pkl"):
    with open("pensieve_memories.pkl", "rb") as f:
        data = pickle.load(f)
        memories = data.get("memories", {})
        topics = data.get("topics", {})
else:
    memories = {}
    topics = {}

# Some hyperparameters
MAX_MEMORIES = 10
MAX_TOPICS = 2
MAX_UNSAVED_MEMORIES = 5
CRYSTALLIZE_INTERVAL = 60  # Seconds (1 minute)

# Flag to control the periodic crystallization
stop_periodic_crystallize = threading.Event()

# implementing the resources and tools
@mcp.tool()
def write_memory(title : str, time_delta: int, text: str, extracted_topics: list[str]):
    """
    Write a memory to the Pensieve.

    Args:
        title (str): The title of the memory.
        time_delta (int): The time delta (in seconds) before the current time.
        text (str): The text of the memory.
        extracted_topics (list[str]): Improtant people, places, or topics in the memory.

    Returns:
        str: Id of the generated memory

    """
    timestamp = time.time() - time_delta
    memory = Memory(title, timestamp, text, extracted_topics, time.time())
    memories[memory.id] = memory

    for topic_name in memory.topics: # Renamed loop variable for clarity
        topic_lower = topic_name.lower() # Use lowercase for dictionary key
        if topic_lower not in topics: # Use lower() for case-insensitive matching
            topics[topic_lower] = Topic(topic_name) # Store Topic object with original name
        # Use the add_memory method of the Topic object
        topics[topic_lower].add_memory(memory.id)

    return f"Memory written successfully with {memory.id}."

@mcp.tool()
def crystalize_memories():
    """
    Crystalizes the memories in the Pensieve into a serialized file (pensieve_memories.pkl).
    
    Returns:
        str: A message indicating the success of the operation and the path to the file.
    """
    file_path = "pensieve_memories.pkl"
    absolute_path = os.path.abspath(file_path)
    # Serialize memories and topics to a file
    try:
        with open(file_path, "wb") as f:
            pickle.dump({"memories": memories, "topics": topics}, f)
        return f"Memories crystalized successfully to {absolute_path}"
    except Exception as e:
        return f"Error crystalizing memories: {e}"

@mcp.tool()
def clear_memories():
    """
    Clears all memories and topics from the Pensieve.
    
    Returns:
        str: A message indicating the success of the operation.
    """
    memories.clear()
    topics.clear()
    crystalize_memories()  # Save the cleared state
    return "All memories cleared successfully."

@mcp.tool()
def get_memories(query: str):
    """
    Retrieves memories relevant to the given query.
    
    Args:
        query (str): The query to search for.
    Returns:
        list: A list of memories relevant to the query.
    """
    all_memories = memories.values()
    query_embedding = qa_model.encode(query)

    # Sort memories based on cosine similarity to query
    sorted_memories = sorted(all_memories, key=lambda m: qa_model.similarity(query_embedding, m.title_embedding), reverse=True)

    # Return the top memories
    return [m.dictionary() for m in sorted_memories[:MAX_MEMORIES]]

@mcp.tool()
def get_topic_timeline(topic: str):
    """
    Retrieves memories related to a specific topic. Memories are then sorted by time. 
    Note that the topic does not need to exactly match the topic name in the Pensieve.
    Semantic similarity will be used to find the most relevant topics and memories will be retrieved accordingly.
    
    Args:
        topic (str): The name of the topic. This can be a person, place, or event.
    Returns:
        list: A list of memories related to the topic.
    """
    relevant_topics = get_similar_topics(topic.lower())  # Get similar topics

    retrieved_memories = []
    for topic in relevant_topics:        
        topic_lower = topic.lower()
        if topic_lower in topics:
            topic_obj = topics[topic_lower]
            memory_ids = topic_obj.memories
            topic_memories = [memories[m_id] for m_id in memory_ids if m_id in memories]
            retrieved_memories.extend(topic_memories)
    retrieved_memories = sorted(retrieved_memories, key=lambda m: m.time)  # Sort by time
    return [m.dictionary() for m in retrieved_memories]

def periodic_crystallize_task():
    """
    Periodically calls crystalize_memories.
    """
    while not stop_periodic_crystallize.is_set():
        crystalize_memories()
        stop_periodic_crystallize.wait(CRYSTALLIZE_INTERVAL)

def get_similar_topics(topic: str):
    if not topics:
        return []

    # Encode the query subject
    topic_embedding = similarity_model.encode(topic)

    # Retrieve existing topic objects and their pre-calculated embeddings
    existing_topic_objects = list(topics.values())
    topic_embeddings = [topic.embedding for topic in existing_topic_objects]
    topic_names = [topic.name for topic in existing_topic_objects] # Get original names

    # Calculate cosine similarities between topic and existing topic embeddings
    similarities = similarity_model.similarity(topic_embedding, topic_embeddings)[0] # Get the first row

    # Pair original topic names with their scores and sort
    scored_topics = sorted(zip(topic_names, similarities), key=lambda item: item[1], reverse=True)

    # Return the names of the MAX_TOPICS topics that are similar to the input topic
    return [name for name, score in scored_topics[:MAX_TOPICS]]

@mcp.resource("memory:://")
def get_all_memory():
    """
    This function returns a dictionary containing the current state of memories and topics.
    
    Returns:
        dict: A dictionary containing the current state of memories and topics.
    """
    return {
        "memories": [m.title for m in memories.values()],
        "topics": [topic.name for topic in topics.values()]
    }
