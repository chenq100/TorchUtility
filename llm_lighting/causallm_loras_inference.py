# cython: language_level=3
### Reference: https://huggingface.co/docs/transformers/main/peft

import torch

from transformers import (
    AutoModelForCausalLM, # A generic model class that will be instantiated as one of the model classes of the library (with a causal language modeling head) 
                          #     when created with the from_pretrained() class method or the from_config() class method.

    AutoTokenizer,        # A generic tokenizer class that will be instantiated as one of the tokenizer classes of the library 
                          #     when created with the AutoTokenizer.from_pretrained() class method.
                          #
    BitsAndBytesConfig    # This is a wrapper class about all possible attributes and features
                          #     that you can play with a model that has been loaded using bitsandbytes. 
                          #
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
    def __init__(self, model_path, model_name):
        self.model_name = model_name

        bnb_config_nf4 = BitsAndBytesConfig(                  # https://huggingface.co/docs/transformers/main_classes/quantization#transformers.BitsAndBytesConfig
                        load_in_4bit=True,                    # enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from bitsandbytes
                        bnb_4bit_quant_type="nf4",            # sets the quantization data type in the bnb.nn.Linear4Bit layers
                        bnb_4bit_use_double_quant=True,       # used for nested quantization where the quantization constants from the first quantization are quantized again
                        bnb_4bit_compute_dtype=torch.bfloat16 # sets the computational type which might be different than the input time
                        )


        # PreTrainedModel -> https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/modeling_utils.py#L1127
        # _BaseAutoModelClass -> https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/auto/auto_factory.py#L400
        #
        #class AutoModelForCausalLM(_BaseAutoModelClass): 
        #    _model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING
        #
        # class PreTrainedModel(nn.Module, ModuleUtilsMixin, GenerationMixin, PushToHubMixin, PeftAdapterMixin):
        # AutoModelForCausalLM.from_pretrained -> https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/auto/auto_factory.py#L443
        # PreTrainedModel.from_pretrained -> https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/modeling_utils.py#L2579
        self.model = AutoModelForCausalLM.from_pretrained(
                # cls, 
                pretrained_model_name_or_path = model_path,
                # *model_args,                                one '*' used to collect all additional positional parameters
                # **kwargs                                    two '**' used to collect all additional keyword arguments
                #
                # 
                # state_dict = kwargs.pop("state_dict", None)
                # from_tf = kwargs.pop("from_tf", False)
                # from_flax = kwargs.pop("from_flax", False)
                # resume_download = kwargs.pop("resume_download", False)
                # proxies = kwargs.pop("proxies", None)
                # output_loading_info = kwargs.pop("output_loading_info", False)
                # use_auth_token = kwargs.pop("use_auth_token", None)
                # trust_remote_code = kwargs.pop("trust_remote_code", None)
                trust_remote_code = True,
                # _ = kwargs.pop("mirror", None)
                # from_pipeline = kwargs.pop("_from_pipeline", None)
                # from_auto_class = kwargs.pop("_from_auto", False)
                # _fast_init = kwargs.pop("_fast_init", True)
                # torch_dtype = kwargs.pop("torch_dtype", None)
                # low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", None)
                # device_map = kwargs.pop("device_map", None)
                device_map="auto",
                # max_memory = kwargs.pop("max_memory", None)
                # offload_folder = kwargs.pop("offload_folder", None)
                # offload_state_dict = kwargs.pop("offload_state_dict", False)
                # load_in_8bit = kwargs.pop("load_in_8bit", False)
                # load_in_4bit = kwargs.pop("load_in_4bit", False)              -> used to enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from bitsandbytes
                # quantization_config = kwargs.pop("quantization_config", None)
                quantization_config = bnb_config_nf4,
                # subfolder = kwargs.pop("subfolder", "")
                # commit_hash = kwargs.pop("_commit_hash", None)
                # variant = kwargs.pop("variant", None)
                # adapter_kwargs = kwargs.pop("adapter_kwargs", {})
                # adapter_name = kwargs.pop("adapter_name", "default")
                # use_flash_attention_2 = kwargs.pop("use_flash_attention_2", False)
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
            self.model.load_adapter( # transformers/src/transformers/integrations/peft.py
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

    def generate(self, text: str, lora_name: str, max_generation_tokens: int):
        if lora_name in self.lora_name_dict:
            self.model.set_adapter( # Sets a specific adapter by forcing the model to use a that adapter and disable the other adapters.
                adapter_name = lora_name # The name of the adapter to set. Can be also a list of strings to set multiple adapters.
                )
        else:
            print(f"Adapter {lora_name} not found. Falling back to using the base model only.")
            # Disable all adapters that are attached to the model. This leads to inferring with the base model only.
            self.model.disable_adapters()

        inputs = self.tokenizer(text, return_tensors="pt")

        device = next(self.model.parameters()).device
        dst_inputs = {key: value.to(device) for key, value in inputs.items()}

        outputs = self.model.generate(**dst_inputs, max_new_tokens = max_generation_tokens)

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response

 




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Causal LLM with LoRA Inference.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model directory.")
    parser.add_argument("--lora", action='append', required=True, help="LoRA parameters in 'name:path' format, can be used multiple times.")
    parser.add_argument("--max_generation_tokens", type=int, required=True, help="Positive integer to control the maximum length of the generation")

    args = parser.parse_args()
    
    clm = CausalLMLoRAsInference(args.model_path, args.model_name)

    for lora_name_and_path in args.lora:
        lora_name, lora_path = lora_name_and_path.split(":", 1)
        clm.load_lora(lora_path = lora_path, lora_name = lora_name, model_name=args.model_name)

    while True:
        lora_name_input = input("Enter LoRA name (or type 'exit' to quit): ")
        if lora_name_input.lower() == 'exit':
            break
        question_input = input(f"Enter your question to {lora_name_input}: ")

        print("Generating response...")
        response = clm.generate(
            text = question_input,
            lora_name = lora_name_input,
            max_generation_tokens = args.max_generation_tokens
            )
        print(f"Generated response: {response}\n")
