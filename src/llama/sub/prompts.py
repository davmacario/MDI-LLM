# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import importlib
import re
from abc import abstractmethod
from json import dumps
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple, Type, Union

import yaml

from .model import Config

if TYPE_CHECKING:
    from . import Tokenizer


class PromptStyle:
    """Base interface for prompt styles."""

    @abstractmethod
    def apply(self, prompt: str, **kwargs: str) -> str:
        return prompt

    def stop_tokens(self, tokenizer: "Tokenizer") -> Tuple[List[int], ...]:
        return ([tokenizer.eos_id],)

    @classmethod
    def from_name(cls, name: str) -> "PromptStyle":
        return prompt_styles[name]()

    @classmethod
    def from_config(cls, config: Config) -> "PromptStyle":
        return model_name_to_prompt_style(config.name)


class Default(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return prompt

    def stop_tokens(self, tokenizer: "Tokenizer") -> Tuple[List[int], ...]:
        return ([tokenizer.eos_id],)


class Llama2FunctionCalling(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        # Has to be before the llama config
        b_func, e_func = "<FUNCTIONS>", "</FUNCTIONS>\n\n"
        b_inst, e_inst = "[INST]", "[/INST]"
        b_sys, e_sys = "<<SYS>>\n", "\n<</SYS>>\n\n"
        # This is an example for how to format functions for the model
        function_metadata = {
            "function": "search_bing",
            "description": (
                "Search the web for content on Bing. This allows users to search online/the internet/the web for"
                " content."
            ),
            "arguments": [{"name": "query", "type": "string", "description": "The search query string"}],
        }

        system_prompt = (
            "You are a helpful, respectful and honest assistant. Always answer as helpfully as"
            "possible. Your only response should be JSON formatted functions"
        )
        # replace the curly braces with double curly braces to escape them
        function_list = dumps(function_metadata).replace("{", "{{").replace("}", "}}")
        return (
            f"{b_func}{function_list.strip()}{e_func}{b_inst}{b_sys}"
            f"{system_prompt.strip()}"
            f"{e_sys}{prompt}{e_inst}\n\n"
        )


class Llama2(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        b_inst, e_inst = "[INST]", "[/INST]"
        b_sys, e_sys = "<<SYS>>\n", "\n<</SYS>>\n\n"
        return (
            f"{b_inst} {b_sys}You are a helpful, respectful and honest assistant. Always answer as helpfully as"
            " possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist,"
            " toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and"
            " positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why"
            " instead of answering something not correct. If you don't know the answer to a question, please don't"
            f" share false information.{e_sys} {prompt} {e_inst} "
        )


class Llama3(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        # https://github.com/meta-llama/llama3/blob/359887376f0aaf30e433f23e25df858d8c2a9833/llama/tokenizer.py#L202-L229
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful assistant.<|eot_id|>\n"  # The system prompt is optional
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def stop_tokens(self, tokenizer: "Tokenizer") -> Tuple[List[int], ...]:
        return (
            [tokenizer.eos_id],
            [tokenizer.token_to_id("<|eot_id|>")],
        )


class CodeLlama(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        # for CodeLLama, we don't set a default system prompt, but it is supported:
        # https://huggingface.co/blog/codellama#conversational-instructions
        # Mistral does not: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1#instruction-format
        b_inst, e_inst = "<s>[INST]", "[/INST]"
        return f"{b_inst} {prompt} {e_inst}"


class TinyLlama(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return (
            "<|system|>\n"
            "You are a friendly chatbot who always gives helpful, detailed, and polite answers.</s>\n"
            "<|user|>\n"
            f"{prompt}</s>\n"
            "<|assistant|>\n"
        )

# Maps prompt style names to PromptStyle classes
prompt_styles: Dict[str, Type[PromptStyle]] = {
    # Model-specific prompt styles
    "llama2-function-calling": Llama2FunctionCalling,
    "llama2": Llama2,
    "codellama": CodeLlama,
    "tinyllama": TinyLlama,
    "llama3": Llama3,
}


def model_name_to_prompt_style(model_name: str) -> PromptStyle:
    if re.search("Llama-2-7b-chat-hf-function-calling-v2", model_name):
        return Llama2FunctionCalling()
    if re.search("Llama-2.*-chat", model_name):
        return Llama2()
    if re.search("Llama-3.*-Instruct", model_name):
        return Llama3()
    if re.search("CodeLlama|Mistral.*Instruct", model_name):
        return CodeLlama()
    if re.search(r"tiny-llama.*chat", model_name):
        return TinyLlama()
    return Default()


def save_prompt_style(style: Union[str, PromptStyle], checkpoint_dir: Path) -> None:
    style = PromptStyle.from_name(style) if isinstance(style, str) else style
    cls = type(style)
    # Allow saving the full module path for user-defined prompt classes
    config = {"class_path": f"{cls.__module__}.{cls.__name__}"}
    with open(checkpoint_dir / "prompt_style.yaml", "w", encoding="utf-8") as file:
        yaml.dump(config, file)


def load_prompt_style(checkpoint_dir: Path) -> PromptStyle:
    with open(checkpoint_dir / "prompt_style.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    # Support loading the full module path for user-defined prompt classes
    full_module_path, cls_name = config["class_path"].rsplit(".", 1)
    module = importlib.import_module(full_module_path)
    cls = getattr(module, cls_name)
    return cls()


def has_prompt_style(checkpoint_dir: Path) -> bool:
    return (checkpoint_dir / "prompt_style.yaml").is_file()

# Util

def get_user_prompt(prompt: str, n_samples: int = 1, **kwargs) -> List[str]:
    """
    Extract the user prompt.

    The given prompt is a string indicating:
        - The prompt itself
        - If starting with 'FILE:', it indicates a text file containing, in each
        paragraph (block of text separated by blank lines) a different prompt

    It is possible to specify the number of samples to return a list with the correct
    length.

    It returns a list containing the prompts as items.
    If the given string is the prompt itself, the list will contain that string repeated
    for n_samples.
    If the file contains more paragraphs than samples, only the first n_samples will be
    considered.
    If the file contains too few, instead, the output list will be padded with "\\n"
    """
    verb = kwargs.get("verb", False)
    supported_ftypes = (".txt", ".md", ".tex")

    if prompt.startswith("FILE:"):
        if not any([prompt.endswith(ext) for ext in supported_ftypes]):
            raise ValueError(
                f"Unsupported file type for {prompt}\nSupported types are: '.txt', '.md'"
            )
        print("Reading prompt(s) from file")
        fname = prompt[5:]
        out = []
        with open(fname, "r") as f:
            curr_sample = ""
            for line in f.readlines():
                if line.strip() != "":
                    curr_sample += line  # Not removing '\n'
                else:  # Paragraph end
                    out.append(curr_sample)
                    curr_sample = ""

                if len(out) == n_samples:
                    break

        if curr_sample != "":
            out.append(curr_sample)

        if len(out) < n_samples:
            out += ["\n"] * (n_samples - len(out))

        assert len(out) == n_samples

        if verb:
            print(out)
        return out
    else:
        return [prompt] * n_samples
