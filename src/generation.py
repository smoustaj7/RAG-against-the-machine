from __future__ import annotations
# pyrefly: ignore [missing-import]
import torch

from typing import Any, List, Optional, cast

from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_MAX_CONTEXT_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 256
_SYSTEM_PROMPT = (
    "You are a code question-answering assistant. Answer the user's "
    "question using ONLY the provided sources. Ground every claim in the "
    "sources. Do NOT invent file paths, line numbers, function names, or "
    "code that is not present in the sources. If the sources do not "
    "contain enough information to answer, say so explicitly."
)

_NO_SOURCES_ANSWER = (
    "No relevant sources were retrieved for this question, so I cannot "
    "provide a grounded answer."
)


def _build_sources_block(
    sources: List[str],
    tokenizer: Any,
    max_tokens: int,
) -> str:
    """Build a token-budgeted ``[Source N]`` block from source texts.
    Sources are added greedily in order; each is included in full only if
    it (plus its header and separators) fits within ``max_tokens``.
    Any source that would overflow the budget is dropped, so the returned
    block never exceeds the requested token budget.
    Args:
        sources: list of retrieved chunk texts.
        tokenizer: a tokenizer exposing ``encode``.
        max_tokens: maximum number of tokens for the whole block.
    Returns:
        A formatted string of sources, or ``""`` if nothing fits.
    """
    if not sources:
        return "(no sources provided)"
    if max_tokens <= 0:
        return ""

    parts: List[str] = []
    used_tokens = 0

    for idx, text in enumerate(sources, start=1):
        header = f"[Source {idx}]"
        header_tokens = len(
            tokenizer.encode(header, add_special_tokens=False)
        )
        remaining = max_tokens - used_tokens
        if remaining <= header_tokens:
            break
        used_tokens += header_tokens

        remaining = max_tokens - used_tokens
        if remaining <= 0:
            break

        full_text_tokens = len(
            tokenizer.encode(text, add_special_tokens=False)
        )
        body = text
        if full_text_tokens > remaining:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            body = tokenizer.decode(
                token_ids[:remaining], skip_special_tokens=True
            )
            used_tokens = max_tokens
        else:
            used_tokens += full_text_tokens

        parts.append(f"{header}\n{body}")

        if used_tokens >= max_tokens:
            break

    return "\n\n".join(parts)


class AnswerGenerator:
    """Generate grounded answers using a HuggingFace causal LM."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        device: Optional[str] = None,
    ) -> None:
        """Load the tokenizer and model.
        Args:
            model_name: HuggingFace model identifier or local path.
            max_context_tokens: token budget for the full prompt.
            max_new_tokens: maximum tokens to generate in the answer.
            device: device to run on (``"cpu"``, ``"cuda"``, etc.).
                Defaults to CUDA if available, else CPU.
        """
        self.model_name = model_name
        self.max_context_tokens = max_context_tokens
        self.max_new_tokens = max_new_tokens

        if device is None:

            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        self._tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto"
        )
        self._model: Any = cast(Any, model)
        self._model.to(self._device)

    def _build_prompt(self, question: str, source_texts: List[str]) -> str:
        """Assemble the chat prompt from sources and question."""
        sources_block = _build_sources_block(
            source_texts, self._tokenizer, self.max_context_tokens
        )
        user_content = (
            f"Question: {question}\n\n"
            f"Sources:\n{sources_block}\n\n"
            "Answer using only the sources above."
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return cast(
            str,
            self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ),
        )

    def generate(
        self,
        question: str,
        source_texts: Optional[List[str]] = None,
    ) -> str:
        """Generate a grounded answer for a question given sources.
        Args:
            question: the natural-language question.
            source_texts: retrieved chunk texts to ground the answer on.
        Returns:
            The generated answer text (or a canned fallback string).
        """
        if not question or not question.strip():
            return "No question provided. Please supply a non-empty question."

        if not source_texts:
            return _NO_SOURCES_ANSWER

        prompt = self._build_prompt(question, source_texts)

        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=self.max_context_tokens,
        ).to(self._device)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        generated = outputs[:, inputs["input_ids"].shape[1]:]
        answer = cast(
            str,
            self._tokenizer.decode(
                generated[0], skip_special_tokens=True
            ),
        ).strip()
        if not answer:
            return _NO_SOURCES_ANSWER
        return answer
