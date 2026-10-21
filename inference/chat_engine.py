"""Chat engine: maintains conversation history and produces responses."""
from __future__ import annotations
from typing import List, Dict, Iterator, Optional

from .generator import Generator


SYSTEM_PROMPT = (
    "You are a helpful, concise AI assistant. "
    "Answer clearly. For math, show your reasoning step by step."
)


class ChatEngine:
    """Wraps a Generator with a chat-style conversation history.

    Conversation is rendered as:
      <|system|>{system}\n<|user|>{u1}\n<|assistant|>{a1}\n<|user|>{u2}\n<|assistant|>
    """

    USER_TAG = "<|user|>"
    ASSISTANT_TAG = "<|assistant|>"
    SYSTEM_TAG = "<|system|>"

    def __init__(self, generator: Generator,
                 system_prompt: str = SYSTEM_PROMPT) -> None:
        self.generator = generator
        self.system_prompt = system_prompt
        self.history: List[Dict[str, str]] = []

    def reset(self) -> None:
        self.history = []

    def _render_prompt(self, user_message: str) -> str:
        parts = [f"{self.SYSTEM_TAG}\n{self.system_prompt}\n"]
        for turn in self.history:
            parts.append(f"{self.USER_TAG}\n{turn['user']}\n")
            parts.append(f"{self.ASSISTANT_TAG}\n{turn['assistant']}\n")
        parts.append(f"{self.USER_TAG}\n{user_message}\n{self.ASSISTANT_TAG}\n")
        return "".join(parts)

    def _trim_history(self, prompt: str) -> str:
        """Drop oldest turns if the encoded prompt exceeds max_seq_len."""
        tokenizer = self.generator.tokenizer
        max_len = self.generator.model.config.max_seq_len - 16  # leave room
        while len(tokenizer.encode(prompt)) > max_len and self.history:
            self.history.pop(0)
            # rebuild last user message from prompt? easier: caller re-renders.
            return self._trim_history(self._render_prompt(
                self.history.pop()["user"] if self.history else ""
            ))
        return prompt

    def chat(self, user_message: str, max_new_tokens: int = 200,
             temperature: float = 0.8, top_k: int = 50, top_p: float = 0.95,
             repetition_penalty: float = 1.15) -> str:
        prompt = self._render_prompt(user_message)
        # Simpler trim: drop oldest turns until it fits
        tokenizer = self.generator.tokenizer
        max_len = self.generator.model.config.max_seq_len - max_new_tokens - 16
        while len(tokenizer.encode(prompt)) > max_len and self.history:
            self.history.pop(0)
            prompt = self._render_prompt(user_message)

        text = self.generator.generate(
            prompt, max_new_tokens=max_new_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
        )
        # Extract assistant content (strip any leaked next-turn tag)
        response = text.split(self.USER_TAG)[0].strip()
        self.history.append({"user": user_message, "assistant": response})
        return response

    def stream(self, user_message: str, max_new_tokens: int = 200,
               temperature: float = 0.8, top_k: int = 50, top_p: float = 0.95,
               repetition_penalty: float = 1.15) -> Iterator[str]:
        prompt = self._render_prompt(user_message)
        tokenizer = self.generator.tokenizer
        max_len = self.generator.model.config.max_seq_len - max_new_tokens - 16
        while len(tokenizer.encode(prompt)) > max_len and self.history:
            self.history.pop(0)
            prompt = self._render_prompt(user_message)

        full = ""
        for piece in self.generator.stream(
            prompt, max_new_tokens=max_new_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
        ):
            full += piece
            if self.USER_TAG in full:
                full = full.split(self.USER_TAG)[0]
                yield full
                break
            yield full
        self.history.append({"user": user_message, "assistant": full.strip()})
