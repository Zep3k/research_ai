"""Provider-independent text with an explicit stable/dynamic boundary."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptContent:
    stable_prefix: str
    dynamic_suffix: str

    def render(self) -> str:
        return self.stable_prefix + self.dynamic_suffix


Prompt = str | PromptContent


def render_prompt(prompt: Prompt) -> str:
    return prompt.render() if isinstance(prompt, PromptContent) else prompt
