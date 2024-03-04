# cython: language_level=3

### Reference: https://huggingface.co/docs/transformers/main/peft

from transformers import (
    AutoModelForCausalLM, # A generic model class that will be instantiated as one of the model classes of the library (with a causal language modeling head) 
                          #     when created with the from_pretrained() class method or the from_config() class method.

    AutoTokenizer         # A generic tokenizer class that will be instantiated as one of the tokenizer classes of the library 
                          #     when created with the AutoTokenizer.from_pretrained() class method.
    )

from peft import (
        LoraConfig,         # class LoraConfig(PeftConfig): Reference: peft/src/peft/tuners/lora/config.py
                            # This is the configuration class to store the configuration of a [`LoraModel`], and inherits from PeftConfig,
                            #     r (`int`): Lora attention dimension (the "rank"). 
                            #     target_modules (`Optional[Union[List[str], str]]`): The names of the modules to apply the adapter to. 
                            #     lora_alpha (`int`): The alpha parameter for Lora scaling.
                            #     lora_dropout (`float`): The dropout probability for Lora layers.
                            #     fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
                            #     bias (`str`): Bias type for LoRA: Can be 'none', 'all' or 'lora_only'. the corresponding biases will be updated during training.
                            #     use_rslora (`bool`): Sets the adapter scaling factor to `lora_alpha/math.sqrt(r)`
                            #     modules_to_save (`List[str]`): List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
                            #     init_lora_weights (`bool` | `Literal["gaussian", "loftq"]`): How to initialize the weights of the adapter layers. 
                            #     layers_to_transform (`Union[List[int], int]`): The layer indices to transform, means that these layers will be embedded in the LoRA matrix.
                            #     layers_pattern (`str`): The layer pattern name, used to further refine or specify the layers selected from layers_to_transform ?
                            #     rank_pattern (`dict`): {"layer name": specific r-value} as opposed to uniform value: 'r'.
                            #     alpha_pattern (`dict`): {"layer name": specific alpha-value} as opposed to uniform value: 'lora_alpha'.
                            #     megatron_config (`Optional[dict]`): The TransformerConfig arguments for Megatron(Nvidia), used to create LoRA's parallel linear layer.
                            #     megatron_core (`Optional[str]`): The core module from Megatron to use, defaults to `"megatron.core"`.
                            #     loftq_config (`Optional[LoftQConfig]`): The configuration of LoftQ, will be used to quantize the backbone weights and initialize Lora layers.
                            #         LoftQ introduces a novel quantization framework tailored for LoRA fine-tuning, 
                            #         effectively bridging the gap between quantized and full-precision models by finding an optimal low-rank initialization, 
                            #         thereby significantly enhancing model generalization and performance on downstream tasks.
        TaskType
        )



lora_configs = {
    'Baichuan2': LoraConfig(                         # https://github.com/baichuan-inc/Baichuan2/blob/main/fine-tune/fine-tune.py#L129
        task_type = TaskType.CAUSAL_LM,
        target_modules = ["W_pack"]
    ),
    'ChatGLM3': LoraConfig(
        task_type = TaskType.CAUSAL_LM,
        target_modules = ["query_key_value"]
    ),

}

class CausalLMLoRAsInference:
    def __init__(self, model_path):
        self.model = AutoModelForCausalLM.from_pretrained( # Reference: 
                pretrained_model_name_or_path = model_path,
                trust_remote_code = True,
                )

        self.tokenizer = AutoTokenizer.from_pretrained(         # Reference: src/transformers/models/auto/tokenization_auto.py
                pretrained_model_name_or_path = model_path,     # A path to a *directory* containing vocabulary files required by the tokenizer.
                #*inputs,                                       # one '*' used to collect all additional positional parameters, 
                                                                #   will be passed along to the Tokenizer `__init__()` method.
                #**kwargs                                       # two '**' used to collect all additional keyword arguments,
                                                                #   will be passed to the Tokenizer `__init__()` method.
                                                                ##### as listed below
                                                                #     config ([`PretrainedConfig`], *optional*)
                                                                #     cache_dir (`str` or `os.PathLike`, *optional*):
                                                                #     force_download (`bool`, *optional*, defaults to `False`):
                                                                #     resume_download (`bool`, *optional*, defaults to `False`):
                                                                #     proxies (`Dict[str, str]`, *optional*):
                                                                #     revision (`str`, *optional*, defaults to `"main"`):
                                                                #     subfolder (`str`, *optional*):
                                                                #     use_fast (`bool`, *optional*, defaults to `True`): Use a [fast Rust-based tokenizer] if it is supported
                                                                #     tokenizer_type (`str`, *optional*):
                                                                #     trust_remote_code (`bool`, *optional*, defaults to `False`): 
                use_fast = False,
                trust_remote_code = True,
                )

        self.tokenizer_type_name = self.tokenizer.__class__.__name__
        print(f"tokenizer_class_name is:  {self.tokenizer_type_name}")

        import types
        from transformers import PreTrainedTokenizerBase        #
        base_pad_method = getattr(PreTrainedTokenizerBase, '_pad')
        tokenizer_pad_method = getattr(self.tokenizer.__class__, '_pad')
        if tokenizer_pad_method is base_pad_method:
            print(f"Derived classes [{self.tokenizer_type_name}] do not override PreTrainedTokenizerBase._pad")
        else:
            print(f"Derived classes [{self.tokenizer_type_name}] override PreTrainedTokenizerBase._pad")

        self.lora_name_dict = {}
    
    def load_lora(self, lora_path: str, lora_name: str, model_name: str) -> None:
        """
        Args:
            lora_path (str):
                The local path where the LoRA adapter exists.
            lora_name (`str`):
                The name of the loaded lora adapter, used as the label to select this LoRA adapter.
            mode_name (str):
                Use as a key in the loraconfig dictionary to search for the LoraConfig, that a derived class of peft.PeftConfig.
        """
        if lora_name not in self.lora_name_dict:
            self.lora_name_dict[lora_name] = [lora_name, model_name]
        else:
            raise ValueError(f"The lora_name key '{lora_name}' already exists in the dictionary.")
        
        try:
            self.model.load_adapter(
                # The identifier of the model to look for on the Hub, or a local path to the saved adapter config file and adapter weights.
                peft_model_id = lora_path,
                # The adapter name to use. If not set, will use the default adapter.
                adapter_name = lora_name,
                # The specific model version to use.
                #revision: Optional[str] = None,
                # Whether to use authentication token to load the remote folder.
                #token: Optional[str] = None,
                # A map that specifies where each submodule should go.
                #device_map: Optional[str] = "auto",
                # A dictionary device identifier to maximum memory, default to the maximum memory available.
                #max_memory: Optional[str] = None,
                # If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
                #offload_folder: Optional[str] = None,
                # `offload_index` argument to be passed to `accelerate.dispatch_model` method.
                #offload_index: Optional[int] = None,
                #  The configuration of the adapter to add, supported adapters are non-prefix tuning and adaption prompts methods.
                #  This argument is used in case users directly pass PEFT state dicts.
                #peft_config: Dict[str, Any] = None,
                # The state dict of the adapter to load. This argument is used in case users directly pass PEFT state dicts. 
                #adapter_state_dict: Optional[Dict[str, "torch.Tensor"]] = None,
                # Additional keyword arguments passed along to the `from_pretrained` method of the adapter config and `find_adapter_config_file` method.
                #adapter_kwargs: Optional[Dict[str, Any]] = None
            )
            print(f"Adapter {lora_name}---{lora_path}, loaded successfully.")
        except Exception as e:
            print(f"Failed to load adapter {lora_name}---{lora_path}, {e}")
            raise

    def generate(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt")

        outputs = self.model.generate(**inputs)

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response

 




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Causal LLM with LoRA Inference.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model directory.")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the LoRA parameters directory.")
    parser.add_argument("--lora_name", type=str, required=True, help="Name of the LoRA parameters to load.")
    parser.add_argument("--question", type=str, required=True, help="Question to ask.")
    args = parser.parse_args()

    clm = CausalLMLoRAsInference(args.model_path)
    clm.load_lora(lora_path=args.lora_path, lora_name=args.lora_path, model_name=args.model_name)
    print(clm.generate(args.question))
