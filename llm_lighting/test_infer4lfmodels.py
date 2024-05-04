
# Adapt the lora-models trained by the llama_factory(v0.5.3) framework for multi-lora inference,
#   refer to: https://github.com/hiyouga/LLaMA-Factory/tree/v0.5.3

from causallm_loras_inference import IChatTokenIDs

from transformers import PreTrainedTokenizer

from llmtuner.data.template import get_template_and_fix_tokenizer

from typing import Literal, List


class ChatTokenIDs4LF(IChatTokenIDs):
    def __init__(self, 
            model_name    # Such as chatglm3 etc.
            ):
        self.model_name = model_name
        self.model_template = None


    def prompt_to(self, tokenizer: PreTrainedTokenizer, prompt: str ) -> List[int]:
        # Reuse llama_factory (v0.5.3) model template source code APIs
        self.model_template = get_template_and_fix_tokenizer(tokenizer, self.model_name)

        messages = [ { "role": "user", "content": prompt } ]

        paired_messages = messages + [{"role": "assistant", "content": ""}]
        prompt, _ = self.model_template.encode_oneturn(
            tokenizer = tokenizer,
            messages = paired_messages,                                  # Append an assistant entry to the messages list
                                                                         #   to indicate it's the assistant's turn to speak in the chat.
            system = None,
            tools = None,
            )
        return prompt


    def to_completion(self, tokenizer: PreTrainedTokenizer, response_tokenids: List[int]) -> str:
        # Refer to llama_factory(v0.5.3) source code implementation
        from dataclasses import dataclass
        @dataclass
        class Response:
            response_text: str
            response_length: int
            finish_reason: Literal["stop", "length"]

        response = tokenizer.batch_decode(response_tokenids, skip_special_tokens = True, clean_up_tokenization_spaces = True)
        results = []
        for i in range(len(response)):
            eos_index = (response_tokenids[i] == tokenizer.eos_token_id).nonzero()
            response_length = (eos_index[0].item() + 1) if len(eos_index) else len(response_tokenids[i])
            results.append(
                Response(
                    response_text = response[i],
                    response_length = response_length,
                    finish_reason = "stop" if len(eos_index) else "length",
                )
            )

        return results[0].response_text.strip()



if __name__ == "__main__":
    from causallm_loras_inference import CausalLMLoRAsInference
    import argparse
    import logging
    parser = argparse.ArgumentParser(description="Run Causal LLM with LoRA Inference.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model directory.")
    parser.add_argument("--lora", action='append', required=True, help="LoRA parameters in 'name:path' format, can be used multiple times.")
    parser.add_argument("--max_generation_tokens", type=int, required=True, help="Positive integer to control the maximum length of the generation")

    args = parser.parse_args()

    clm = CausalLMLoRAsInference(args.model_path, args.model_name, q_bits = 4, device_map = "cuda", log_level = logging.INFO)

    for lora_name_and_path in args.lora:
        lora_name, lora_path = lora_name_and_path.split(":", 1)
        clm.load_lora(lora_path = lora_path, lora_name = lora_name, model_name=args.model_name)

    while True:
        lora_name_input = input("Enter LoRA name (or type 'exit' to quit): ")
        if lora_name_input.lower() == 'exit':
            break
        prompt_filename_input = input(f"Enter your prompt_filename to {lora_name_input}: ")
        prompt = None
        with open(prompt_filename_input, "r", encoding='utf-8') as file:
            prompt = file.read()

        print("Making chat response...")
        response = clm.chat(
            prompt = prompt,
            lora_chat_tokenids = ChatTokenIDs4LF(model_name=args.model_name),
            lora_or_prefix_name = lora_name_input,
            max_new_tokens = args.max_generation_tokens
            )
        print(f"{response}") 
