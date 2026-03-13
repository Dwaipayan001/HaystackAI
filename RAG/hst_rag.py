from datasets import load_dataset
from haystack import Document
from haystack.components.embedders import HuggingFaceAPIDocumentEmbedder, HuggingFaceAPITextEmbedder
from haystack_integrations.components.retrievers.faiss import FAISSEmbeddingRetriever
from haystack_integrations.document_stores.faiss import FAISSDocumentStore
from dotenv import load_dotenv
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.nvidia import NvidiaChatGenerator
import os
from haystack import Pipeline
from haystack.utils import Secret
import warnings

warnings.filterwarnings("ignore")



load_dotenv()

document_store = FAISSDocumentStore(index_path="./hst_rag_index",embedding_dim=1024)
embedding = HuggingFaceAPIDocumentEmbedder(
    api_type="serverless_inference_api",
    api_params={"model":"Snowflake/snowflake-arctic-embed-l-v2.0"},
    token=Secret.from_token(os.getenv("HF_TOKEN"))
)

dataset = load_dataset("bilgeyucel/seven-wonders",split="train")
docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

docs_with_embeddings = embedding.run(docs)
document_store.write_documents(docs_with_embeddings['documents'])

text_embed = HuggingFaceAPITextEmbedder(api_type="serverless_inference_api",
    api_params={"model":"Snowflake/snowflake-arctic-embed-l-v2.0"},
    token=Secret.from_token(os.getenv("HF_TOKEN")))

retriever = FAISSEmbeddingRetriever(document_store=document_store)

template = [
    ChatMessage.from_user(
        """
        Given the following information, answer the question.

        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}

        Question: {{question}}
        Answer:

        """
    )
]

prompt_builder = ChatPromptBuilder(template=template)
chat_generator = NvidiaChatGenerator(
    api_key=Secret.from_token(token=os.getenv("NVIDIA_API_KEY")),
    model="nvidia/nemotron-3-super-120b-a12b",
)

rag_pipeline = Pipeline()
rag_pipeline.add_component("text_embedder",text_embed)
rag_pipeline.add_component("retriever",retriever)
rag_pipeline.add_component("prompt_builder",prompt_builder)
rag_pipeline.add_component("llm",chat_generator)

rag_pipeline.connect("text_embedder.embedding","retriever.query_embedding")
rag_pipeline.connect("retriever","prompt_builder")
rag_pipeline.connect("prompt_builder.prompt","llm.messages")

question = "What does Rhodes Statue look like?"

response = rag_pipeline.run({"text_embedder": {"text": question}, "prompt_builder": {"question": question}})

print(response["llm"]["replies"][0].text)