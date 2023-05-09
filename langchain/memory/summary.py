from typing import Any, Dict, List, Type

from pydantic import BaseModel, root_validator

from langchain.chains.llm import LLMChain
from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.prompt import SUMMARY_PROMPT
from langchain.prompts.base import BasePromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain.schema import (
    BaseMessage,
    SystemMessage,
    get_buffer_string,
)

from typing import Optional
import redis


class SummarizerMixin(BaseModel):
    human_prefix: Optional[str] = None
    ai_prefix: Optional[str] = None
    llm: BaseLanguageModel
    prompt: BasePromptTemplate = SUMMARY_PROMPT
    summary_message_cls: Type[BaseMessage] = SystemMessage

    def __init__(
        self,
        human_name: str,
        bot_name: str,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.human_prefix = human_name
        self.ai_prefix = bot_name

    def predict_new_summary(
        self, messages: List[BaseMessage], existing_summary: str
    ) -> str:
        new_lines = get_buffer_string(
            messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

        chain = LLMChain(llm=self.llm, prompt=self.prompt)
        return chain.predict(summary=existing_summary, new_lines=new_lines)


class ConversationSummaryMemory(BaseChatMemory, SummarizerMixin):
    """Conversation summarizer to memory."""

    buffer: str = ""
    usrNumber: str
    memory_key: str = "history"  #: :meta private:
    redis_client: Optional[redis.StrictRedis] = None

    def __init__(self, usrNumber, redis_client: redis.StrictRedis, *args: Any, **kwargs: Any):
        super().__init__(usrNumber=usrNumber, *args, **kwargs)  # Add usrNumber to the arguments
        self.redis_client = redis_client
        self.usrNumber = usrNumber

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        self.buffer = self.redis_client.hget(self.usrNumber, self.memory_key) or ""
        if self.return_messages:
            buffer: Any = [self.summary_message_cls(content=self.buffer)]
        else:
            buffer = self.buffer
        return {self.memory_key: buffer}

    @root_validator()
    def validate_prompt_input_variables(cls, values: Dict) -> Dict:
        """Validate that prompt input variables are consistent."""
        prompt_variables = values["prompt"].input_variables
        expected_keys = {"summary", "new_lines"}
        if expected_keys != set(prompt_variables):
            raise ValueError(
                "Got unexpected prompt input variables. The prompt expects "
                f"{prompt_variables}, but it should have {expected_keys}."
            )
        return values

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        super().save_context(inputs, outputs)
        self.buffer = self.predict_new_summary(
            self.chat_memory.messages[-2:], self.buffer
        )
        self.redis_client.hset(self.usrNumber, self.memory_key, self.buffer)

    def clear(self) -> None:
        """Clear memory contents."""
        super().clear()
        self.redis_client.hdel(self.usrNumber, self.memory_key)
        self.buffer = ""