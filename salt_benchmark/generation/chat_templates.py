from __future__ import annotations

DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."


def hf_chat_format(instructions: list[str], default_system: str = DEFAULT_SYSTEM_PROMPT, tokenizer=None) -> list[list[dict[str, str]]]:
    """Build HuggingFace-style chat messages for plain instruction strings.

    Args:
        instructions: User instructions to place in chat format.
        default_system: System message used when the tokenizer template supports it.
        tokenizer: Optional tokenizer whose ``chat_template`` determines system support.

    Returns:
        One chat-message list per instruction.
    """
    chat_template = getattr(tokenizer, "chat_template", None) if tokenizer is not None else None
    include_system = isinstance(chat_template, str) and "system" in chat_template.lower()
    chats: list[list[dict[str, str]]] = []
    for instruction in instructions:
        chat = []
        if include_system:
            chat.append({"role": "system", "content": default_system})
        chat.append({"role": "user", "content": instruction})
        chats.append(chat)
    return chats


def render_hf_prompts(tokenizer, instructions: list[str], default_system: str = DEFAULT_SYSTEM_PROMPT) -> list[str]:
    """Render instruction strings with a tokenizer chat template when available.

    If system-role rendering fails, the function retries with a user-only chat.
    If templating still fails or no template exists, it returns the raw instruction.
    """
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template.strip():
        return instructions

    rendered_prompts: list[str] = []
    for instruction, chat in zip(instructions, hf_chat_format(instructions, default_system=default_system, tokenizer=tokenizer)):
        try:
            rendered_prompts.append(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
            continue
        except Exception:
            pass

        no_system_chat = [{"role": "user", "content": instruction}]
        try:
            rendered_prompts.append(tokenizer.apply_chat_template(no_system_chat, tokenize=False, add_generation_prompt=True))
        except Exception:
            rendered_prompts.append(instruction)
    return rendered_prompts
