from typing import Any, Dict, List

from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema import BaseMessage, get_buffer_string

import logging
from abc import ABC, abstractmethod
from itertools import islice
from typing import Any, Dict, Iterable, List, Optional
from dotenv import load_dotenv
from pathlib import Path



from pydantic import Field

from langchain.chains.llm import LLMChain
from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.prompt import (
    ENTITY_EXTRACTION_PROMPT,
    ENTITY_SUMMARIZATION_PROMPT,
)

from langchain.memory.utils import get_prompt_input_key
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import BaseMessage, get_buffer_string
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.base_language import BaseLanguageModel


import pinecone
import os
import sys
import redis
import asyncio
import threading




logger = logging.getLogger(__name__)

class BaseEntityStore(ABC):
    @abstractmethod
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get entity value from store."""
        pass

    @abstractmethod
    def set(self, key: str, value: Optional[str]) -> None:
        """Set entity value in store."""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete entity value from store."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if entity exists in store."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Delete all entities from store."""
        pass




class PineconeEntityStore(BaseEntityStore):
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        index_name: str = "test-index",
        namespace: Optional[str] = None,
        embeddings: OpenAIEmbeddings = None,  # Change this to None
        usrNumber: Optional[str] = None,
        redis_client: Optional[redis.StrictRedis] = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        pinecone.init(api_key=api_key, environment=environment)
        self.index = pinecone.Index(index_name=index_name)
        self.embeddings = embeddings or OpenAIEmbeddings(openai_api_key=openai_api_key)  # Pass the API key here
        self.namespace = namespace

        if usrNumber is not None and redis_client is not None:
            self.redis_cache = RedisCache(usrNumber, redis_client=redis_client)
        else:
            self.redis_cache = None

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        # Check for cached value first
        if self.redis_cache:
            cached_value = self.redis_cache.get(key)
            if cached_value is not None:
                print(f"Using cached result for key '{key}': {cached_value}")
                return cached_value

        # Retrieve from Pinecone and cache if not found in Redis
        query_embedding = self.embeddings.embed_query(key)
        query_response = self.index.query(
            top_k=5,
            include_values=True,
            include_metadata=True,
            vector=query_embedding,
            namespace=self.namespace,
        )

        query_result = query_response.matches
        for result in query_result:
            value = query_result[0].metadata.get("metadata", default)
            print(f"Found result for key '{key}': {value}")

            # Cache the result
            if self.redis_cache:
                self.redis_cache.set(key, value)

            return value
        print(f"No result found for key '{key}'")
        return default

    def set(self, key: str, value: Optional[str]) -> None:
        if not value:
            return self.delete(key)

        if "new information" in value.lower():
            print(f"Skipping upsert for key '{key}' as value contains 'new information': {value}")
            return

        entity_embedding = self.embeddings.embed_query(value)
        self.index.upsert(vectors=[(key, entity_embedding, {"metadata": value})], namespace=self.namespace)
        print(f"Upserted value '{value}' for key '{key}'")


    def delete(self, key: str) -> None:
        self.index.delete(ids=[key], namespace=self.namespace)
        print(f"Deleted key '{key}'")

    def exists(self, key: str) -> bool:
        query_embedding = self.embeddings.embed_query(key)
        query_response = self.index.query(
            top_k=5,
            include_values=False,
            include_metadata=False,
            vector=query_embedding,
            namespace=self.namespace,
        )

        query_result = query_response.matches
        print(f"Key '{key}' exists: {bool(query_result)}")
        return bool(query_result)

    def clear(self) -> None:
        self.index.delete_all(namespace=self.namespace)
        print("Cleared all entries")

    def batch_upsert(self, data: List[tuple[str, str]]) -> None:
        vectors = []
        for key, value in data:
            if "new information" in value.lower():
                print(f"Skipping upsert for key '{key}' as value contains 'new information': {value}")
                continue
            entity_embedding = self.embeddings.embed_query(value)
            vectors.append((key, entity_embedding, {"metadata": value}))
        if vectors:
            self.index.upsert(vectors=vectors, namespace=self.namespace)
            print(f"Upserted {len(vectors)} values")
        
class RedisCache:
    def __init__(self, usrNumber: str, redis_client: redis.StrictRedis):
        self.client = redis_client
        self.usrNumber = usrNumber

    def get(self, key: str) -> Optional[str]:
        cache_key = f"{self.usrNumber}:{key}"
        value = self.client.get(cache_key)
        if value is not None:
            return value.decode()
        return None

    def set(self, key: str, value: str, ttl: int = 300) -> None:
        cache_key = f"{self.usrNumber}:{key}"
        self.client.set(cache_key, value, ex=ttl)

    def delete(self, key: str) -> None:
        cache_key = f"{self.usrNumber}:{key}"
        self.client.delete(cache_key)


class ConversationEntityCache(BaseChatMemory):
    """Buffer for storing conversation memory."""

    human_prefix: Optional[str] = None
    ai_prefix: Optional[str] = None
    openai_api_key: Optional[str] = None
    api_key: Optional[str] = None
    environment: Optional[str] = None
    llm: BaseLanguageModel
    entity_extraction_prompt: BasePromptTemplate = ENTITY_EXTRACTION_PROMPT
    entity_summarization_prompt: BasePromptTemplate = ENTITY_SUMMARIZATION_PROMPT
    entity_cache: List[str] = []    
    entity_store: BaseEntityStore = Field(default_factory=PineconeEntityStore)
    entity_updates_queue: List[tuple[str, str]] = []


    memory_key: str = "history"  #: :meta private:
    k: int = 1

    def __init__(
        self,
        human_name: str,
        bot_name: str,
        usrNumber: Optional[str] = None,  # Add this argument
        redis_client: Optional[redis.StrictRedis] = None,  # Add this argument
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.human_prefix = human_name
        self.ai_prefix = bot_name  

        # Update the PineconeEntityStore with the provided namespace
        if usrNumber is not None:
            self.entity_store.namespace = usrNumber
        if redis_client is not None:
            self.entity_store.redis_cache = RedisCache(usrNumber, redis_client)


        self.start_async_queue_processing()

    def start_async_queue_processing(self):
        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.process_entity_updates_queue())
            loop.close()

        thread = threading.Thread(target=run_async)
        thread.start()

    async def process_entity_updates_queue(self):
        while True:
            if self.entity_updates_queue:
                logging.info("Processing entity_updates_queue")
                self.entity_store.batch_upsert(self.entity_updates_queue)
                self.entity_updates_queue = []
            await asyncio.sleep(60)  # Adjust the time between processing as needed

    @property
    def buffer(self) -> List[BaseMessage]:
        """String buffer of memory."""
        return self.chat_memory.messages

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.
        :meta private:
        """
        return [self.memory_key]



    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        chain = LLMChain(llm=self.llm, prompt=self.entity_extraction_prompt)
        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        buffer_string = get_buffer_string(
            self.buffer[-self.k * 2 :],
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )
        output = chain.predict(
            history=buffer_string,
            input=inputs[prompt_input_key],
        )
        if output.strip() == "NONE":
            entities = []
        else:
            entities = [w.strip() for w in output.split(",")]
        entity_summaries = {}
        for entity in entities:
            entity_summaries[entity] = self.entity_store.get(entity, "")
        self.entity_cache = entities

        if self.return_messages:
            buffer: Any = self.buffer[-self.k * 2 :]
        else:
            buffer = buffer_string
        return {
            self.memory_key: buffer,
            "entities": entity_summaries,
        }

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        super().save_context(inputs, outputs)

        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key

        buffer_string = get_buffer_string(
            self.buffer[-self.k * 2 :],
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )
        input_data = inputs[prompt_input_key]
        chain = LLMChain(llm=self.llm, prompt=self.entity_summarization_prompt)

        for entity in self.entity_cache:
            existing_summary = self.entity_store.get(entity, "")
            output = chain.predict(
                summary=existing_summary,
                entity=entity,
                history=buffer_string,
                input=input_data,
            )
            self.entity_updates_queue.append((entity, output.strip()))
            print(f"Stored vector for '{entity}' for upserting later")  # Print statement to show vector storage


    def clear(self) -> None:
        """Clear memory contents."""
        self.chat_memory.clear()
        self.entity_store.clear()
