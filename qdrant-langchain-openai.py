from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from qdrant_client.models import Distance, VectorParams
import qdrant_client
import os
import dotenv

dotenv.load_dotenv()

# QDRANT_HOST=https://06145fee46df.us-east-1-0.aws.cloud.qdrant.io
# QDRANT_API_KEY=
# QDRANT_COLLECTION=krishna-db
# OPENAI_API_KEY=

# Using local storage
# from qdrant_client import QdrantClient
# client = qdrant_client.QdrantClient(":memory:")
# or
# client = qdrant_client.QdrantClient(path="./local_db")  # Persists changes to disk

# Using Qdrant server
# from qdrant_client import QdrantClient
# client = QdrantClient(host="localhost", port=6333)
# # or
# client = QdrantClient(url="http://localhost:6333")
#

# Using cloud
client = qdrant_client.QdrantClient(
    url=os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY")
)

collection_config = VectorParams(
    size=1536,
    distance=Distance.COSINE
)

# create collection
client.recreate_collection(
    collection_name=os.getenv("QDRANT_COLLECTION"),
    vectors_config=collection_config
)

embeddings = OpenAIEmbeddings()

# create your vector store
vectorstore = Qdrant(
    client=client,
    collection_name=os.getenv("QDRANT_COLLECTION"),
    embeddings=embeddings
)


# add documents to your vector database
def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


with open("article.txt") as f:
    raw_text = f.read()

texts = get_chunks(raw_text)
print("CHUNK ---> ", texts)

# add to vector store
vectorstore.add_texts(texts)

# plug the vector store to your retrieval chain
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

query = "What are the reason of rise in right wing parties or leaders in europe?"
response = qa.run(query)

print(response)
