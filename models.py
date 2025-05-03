import uuid
from sentence_transformers import SentenceTransformer # Added import

qa_model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

class Memory:
    def __init__(self, title: str, time: int, text: str, topics: list[str]):
        self.id = id(uuid.uuid4())
        self.title = title
        self.time = time
        self.text = text
        self.topics = topics
        # Generate and store the title embedding
        self.title_embedding = qa_model.encode(title)

    def dictionary(self):
        return {
            "id": self.id,
            "title": self.title,
            "time": self.time,
            "text": self.text,
            "topics": self.topics,
        }

class Topic:
    def __init__(self, name: str):
        self.name = name
        self.embedding = similarity_model.encode(name)
        self.memories = []

    def add_memory(self, memory_id: str):
        self.memories.append(memory_id)