# cython: language_level=3
### Reference: https://huggingface.co/docs/transformers/main/peft

from transformers import (
    AutoModelForCausalLM,         # A generic model class that will be instantiated as one of the model classes of the library (with a causal language modeling head) 
                                  #     when created with the from_pretrained() class method or the from_config() class method.

    AutoTokenizer,                # A generic tokenizer class that will be instantiated as one of the tokenizer classes of the library 
                                  #     when created with the AutoTokenizer.from_pretrained() class method.
                                  #
    PreTrainedTokenizer,          #
                                  #
    BitsAndBytesConfig,           # This is a wrapper class about all possible attributes and features
                                  #     that you can play with a model that has been loaded using bitsandbytes. 
                                  #
    LogitsProcessorList,          # transformers/src/transformers/generation/logits_process.py
                                  #   Create a list of [`LogitsProcessor`] or [`LogitsWarper`] to subsequently process a `scores` input tensor,
                                  #     the 'scores' tensor represents the model's output predictions for each possible next token, based on the input sequence,
                                  #     these scores are used to determine the likelihood of each token being the next in the generated sequence.
    InfNanRemoveLogitsProcessor,  # ...
                                  #
    GenerationConfig,             # transformers/src/transformers/generation/configuration_utils.py
                                  # Class that holds a configuration for a generation task.

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


import torch

from typing import (
        List,
        Dict,
        Sequence,
        Tuple,
        Union,
        Literal,
        Optional,
        )

from abc import ABC, abstractmethod



class IChatTokenIDs(ABC):
    """
    This abstract base class is designed for the transformation process involved in chat interactions with a LLM:
    prompt --> messages --> input token ids --> model.generate --> output token ids --> response token ids --> completion.
    
    It provides a structured approach to convert textual chat prompts into a format that is suitable for LLM processing and 
    to interpret the model's output back into human-readable text responses.
    
    The process involves:
    - Converting the initial prompt into a structured 'messages' format.
    - Transforming these messages into input token IDs for the model.
    - Generating output token IDs from the model based on the input token IDs.
    - Extracting response token IDs from the model's output.
    - Transforming these response token IDs back into a completion text.

    Methods:
    - prompt_to(tokenizer, prompt): Converts a text prompt into a list of input token IDs.
    - to_completion(tokenizer, response_tokenids): Converts a list of token IDs back into text.

    The class is abstract and requires concrete implementations of its methods to specify the exact transformation logic.
    """
    def __init__(self, config: Dict[str, any] = None):
        """
        Initialize the instance with optional configuration settings.

        Parameters:
        - config: A dictionary containing optional configuration options for the transformation process.
                  These options can customize how prompts and responses are processed.
        """
        self.config = config

    def update_config(self, config: Dict[str, any]) -> None:
        """
        Update the instance's configuration settings.

        This method allows updating the transformation process's configuration options at runtime.

        Parameters:
        - config: A dictionary containing new configuration options to update the transformation process.
        """
        self.config = config

    @abstractmethod
    def prompt_to(self, tokenizer: PreTrainedTokenizer, prompt: str) -> List[int]:
        """
        Transform a text prompt into a list of token IDs suitable for LLM processing, using the provided tokenizer and current configuration settings.

        This method is intended to prepare user prompts for processing by a large language model (LLM).

        Parameters:
        - tokenizer: An instance of PreTrainedTokenizer, used for converting the text prompt into token IDs.
        - prompt: The text prompt to be transformed.

        Returns:
        A list of token IDs generated from the prompt, ready for input into an LLM.
        """
        pass

    @abstractmethod
    def to_completion(self, tokenizer: PreTrainedTokenizer, response_tokenids: List[int]) -> str:
        """
        Transform a list of response token IDs back into human-readable text (completion) using the provided tokenizer and current configuration settings.

        This method is intended to convert the output of a large language model (LLM), represented as token IDs, back into readable text.

        Parameters:
        - tokenizer: An instance of PreTrainedTokenizer, used for converting token IDs back into text.
        - response_tokenids: A list of token IDs representing the model's response.

        Returns:
        A human-readable text string generated from the list of response token IDs.
        """
        pass





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

        self.default_logits_processor = LogitsProcessorList().append(InfNanRemoveLogitsProcessor())
    
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

    @torch.inference_mode()
    def generate(self, text: str, lora_name: str, max_generation_tokens: int, 
                    logits_processor: LogitsProcessorList  = None,
            ):
        if lora_name in self.lora_name_dict:
            self.model.set_adapter( # Sets a specific adapter by forcing the model to use a that adapter and disable the other adapters.
                adapter_name = lora_name # The name of the adapter to set. Can be also a list of strings to set multiple adapters.
                )
        else:
            print(f"Adapter {lora_name} not found. Falling back to using the base model only.")
            # Disable all adapters that are attached to the model. This leads to inferring with the base model only.
            self.model.disable_adapters()

        # the return inputs is a dict like: {'input_ids': tensor([[64790, 64792, 24954]]), 
        #                                    'attention_mask': tensor([[1, 1, 1]]), 
        #                                    'position_ids': tensor([[0, 1, 2]])
        #                                   }
        inputs = self.tokenizer(text, return_tensors="pt")

        device = next(self.model.parameters()).device
        dst_inputs = {key: value.to(device) for key, value in inputs.items()}

        outputs = self.model.generate( # class GenerationMixin -> transformers/src/transformers/generation/utils.py
                # inputs: Optional[torch.Tensor] = None, -> The sequence used as a prompt for the generation or as model inputs to the encoder.
                input_ids = dst_inputs.get('input_ids', None),
                # generation_config: Optional[GenerationConfig] = None, -> The generation configuration to be used as base parametrization for the generation call.
                # logits_processor: Optional[LogitsProcessorList] = None, -> a
                logits_processor = logits_processor if logits_processor is not None else self.default_logits_processor,
                # stopping_criteria: Optional[StoppingCriteriaList] = None,
                # prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
                # synced_gpus: Optional[bool] = None,
                # assistant_model: Optional["PreTrainedModel"] = None,
                # streamer: Optional["BaseStreamer"] = None,
                # negative_prompt_ids: Optional[torch.Tensor] = None,
                # negative_prompt_attention_mask: Optional[torch.Tensor] = None,
                # **kwargs,
                attention_mask = dst_inputs.get('attention_mask', None),
                position_ids = dst_inputs.get('position_ids', None),
                max_new_tokens = max_generation_tokens
                )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response

    @torch.inference_mode()
    def chat(self, prompt: str, 
                   chat_tokenids: IChatTokenIDs,
                   lora_name: str, 
                   max_generation_tokens: int,
                   logits_processor: LogitsProcessorList  = None,
            ) -> str:

        """
        Conducts a chat session using a specified TokenIDs's transformer and configurations.

        Args:
            prompt (str): 

            chat_tokenids (IChatTokenIDs): 
                An instance of IChatTokenIDs used for:
                    prompt_to: transforming prompt into input token IDs suitable for LLM chating.
                    to_completion: transforming generated chat rresponse_tokenids into completion.
        
            lora_name (str): 
                The name of the LLM to be used for generating responses. This could specify a particular model version or configuration.
        
            max_generation_tokens (int): 
                The maximum number of tokens to be generated for the response. This limits the length of the model's output.
        
            logits_processor (Optional[LogitsProcessorList]): 
                An optional list of logits processors to apply to the logits before the softmax step, allowing for manipulation of the logits to control the generation process.
        """
        prompt_ids = chat_tokenids.prompt_to(self.tokenizer, prompt)
        prompt_length = len(prompt_ids)
        input_ids = torch.tensor([prompt_ids], device = next(self.model.parameters()).device)

        eos_token_ids = [self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids
        generation_config = GenerationConfig(                      # A large number of these flags control the logits or the stopping criteria of the generation.
                                                                   # Parameters that control the length of the output
                # max_length (`int`, *optional*, defaults to 20): -> The maximum length the generated tokens can have.  
                # max_new_tokens (`int`, *optional*):             -> The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
                # min_length (`int`, *optional*, defaults to 0):  -> The minimum length of the sequence to be generated: the length of the input prompt + `min_new_tokens`.
                # min_new_tokens (`int`, *optional*):             -> The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt.
                # early_stopping (`bool` or `str`, *optional*, defaults to `False`): -> Controls the stopping condition for beam-based methods, like beam-search.
                # max_time(`float`, *optional*):                  -> The maximum amount of time you allow the computation to run for in seconds.
                                                                   # Parameters that control the generation strategy used
                # do_sample (`bool`, *optional*, defaults to `False`): -> Whether or not to use sampling ; use greedy decoding otherwise.
                # num_beams (`int`, *optional*, defaults to 1):   -> Number of beams for beam search. 1 means no beam search.
                # num_beam_groups (`int`, *optional*, defaults to 1): -> Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
                # penalty_alpha (`float`, *optional*):            -> The values balance the model confidence and the degeneration penalty in contrastive search decoding.
                # use_cache (`bool`, *optional*, defaults to `True`): -> Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
                                                                   # Parameters for manipulation of the model output logits 
                # temperature (`float`, *optional*, defaults to 1.0): -> The value used to modulate the next token probabilities.
                # top_k (`int`, *optional*, defaults to 50):      -> The number of highest probability vocabulary tokens to keep for top-k-filtering.
                # top_p (`float`, *optional*, defaults to 1.0):   -> If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation.
                # typical_p (`float`, *optional*, defaults to 1.0):
                # epsilon_cutoff (`float`, *optional*, defaults to 0.0): -> If set to float strictly between 0 and 1, only tokens with a conditional probability greater than `epsilon_cutoff` will be sampled.
                # eta_cutoff (`float`, *optional*, defaults to 0.0): -> Eta sampling is a hybrid of locally typical sampling and epsilon sampling.
                # diversity_penalty (`float`, *optional*, defaults to 0.0): -> This value is subtracted from a beam's score if it generates a token same as any beam from other group at a particular time.
                # repetition_penalty (`float`, *optional*, defaults to 1.0): -> The parameter for repetition penalty. 1.0 means no penalty.
                # encoder_repetition_penalty (`float`, *optional*, defaults to 1.0): -> The paramater for encoder_repetition_penalty.
                # length_penalty (`float`, *optional*, defaults to 1.0): -> Exponential penalty to the length that is used with beam-based generation.
                # no_repeat_ngram_size (`int`, *optional*, defaults to 0): -> If set to int > 0, all ngrams of that size can only occur once.
                # bad_words_ids(`List[List[int]]`, *optional*):   ->  List of list of token ids that are not allowed to be generated.
                # force_words_ids(`List[List[int]]` or `List[List[List[int]]]`, *optional*): -> List of token ids that must be generated.
                # renormalize_logits (`bool`, *optional*, defaults to `False`): -> Whether to renormalize the logits after applying all the logits processors or warpers (including the custom ones).
                # constraints (`List[Constraint]`, *optional*):   -> Custom constraints that can be added to the generation to ensure that the output will contain the use of certain tokens as defined by `Constraint` objects.
                # forced_bos_token_id (`int`, *optional*, defaults to `model.config.forced_bos_token_id`): -> The id of the token to force as the first generated token after the `decoder_start_token_id`.
                # forced_eos_token_id (`Union[int, List[int]]`, *optional*, defaults to `model.config.forced_eos_token_id`): -> The id of the token to force as the last generated token when `max_length` is reached.
                # remove_invalid_values (`bool`, *optional*, defaults to `model.config.remove_invalid_values`): -> Whether to remove possible *nan* and *inf* outputs of the model to prevent the generation method to crash.
                # exponential_decay_length_penalty (`tuple(int, float)`, *optional*): -> This Tuple adds an exponentially increasing length penalty, after a certain amount of tokens have been generated.
                # suppress_tokens  (`List[int]`, *optional*):     -> A list of tokens that will be suppressed at generation.
                # begin_suppress_tokens  (`List[int]`, *optional*): -> A list of tokens that will be suppressed at the beginning of the generation.
                # forced_decoder_ids (`List[List[int]]`, *optional*): -> A list of pairs of integers which indicates a mapping from generation indices to token indices that will be forced before sampling.
                # sequence_bias (`Dict[Tuple[int], float]`, *optional*)): -> Dictionary that maps a sequence of tokens to its bias term.
                # guidance_scale (`float`, *optional*):           -> Higher guidance scale encourages the model to generate samples that are more closely linked to the input prompt.
                # low_memory (`bool`, *optional*):                -> Switch to sequential beam search and sequential topk for contrastive search to reduce peak memory.
                                                                  # Parameters that define the output variables of `generate`
                # num_return_sequences(`int`, *optional*, defaults to 1): -> The number of independently computed returned sequences for each element in the batch.
                # output_attentions (`bool`, *optional*, defaults to `False`): -> Whether or not to return the attentions tensors of all attention layers.
                # output_hidden_states (`bool`, *optional*, defaults to `False`): -> Whether or not to return the hidden states of all layers.
                # output_scores (`bool`, *optional*, defaults to `False`): -> Whether or not to return the prediction scores.
                # return_dict_in_generate (`bool`, *optional*, defaults to `False`): -> Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
                                                                  # Special tokens that can be used at generation time
                # pad_token_id (`int`, *optional*):               -> The id of the *padding* token.
                pad_token_id = self.tokenizer.pad_token_id,
                # bos_token_id (`int`, *optional*):               -> The id of the *beginning-of-sequence* token.
                # eos_token_id (`Union[int, List[int]]`, *optional*): -> The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
                eos_token_id = eos_token_ids,
                                                                  # Generation parameters exclusive to encoder-decoder models
                # encoder_no_repeat_ngram_size (`int`, *optional*, defaults to 0):
                # decoder_start_token_id (`int`, *optional*):
                                                                  # Generation parameters exclusive to [assistant generation](https://arxiv.org/abs/2211.17192)
                # num_assistant_tokens (`int`, *optional*, defaults to 5): 
                # num_assistant_tokens_schedule (`str`, *optional*, defaults to `"heuristic"`):



                # generation_kwargs: Additional generation kwargs will be forwarded to the `generate` function of the model.
                )
        
        if lora_name in self.lora_name_dict:
            self.model.set_adapter( # Sets a specific adapter by forcing the model to use a that adapter and disable the other adapters.
                adapter_name = lora_name # The name of the adapter to set. Can be also a list of strings to set multiple adapters.
                )
        else:
            print(f"Adapter {lora_name} not found. Falling back to using the base model only.")
            # Disable all adapters that are attached to the model. This leads to inferring with the base model only.
            self.model.disable_adapters()

        generated_tokenids = self.model.generate(
                input_ids = input_ids,
                generation_config = generation_config,
                logits_processor = logits_processor if logits_processor is not None else self.default_logits_processor,
                max_new_tokens = max_generation_tokens,
                )
        
        response_tokenids = generated_tokenids[:, prompt_length:]

        completion = chat_tokenids.to_completion(self.tokenizer, response_tokenids)

        return completion




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
