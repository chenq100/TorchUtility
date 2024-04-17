# cython: language_level=3
### Reference: https://huggingface.co/docs/transformers/main/peft

from transformers import (
    AutoConfig,                   # 
                                  #
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
    QuantoConfig,                 # This is a wrapper class about all possible attributes and features that you can play with a model that has been loaded using `quanto`.
                                  # transformers-4.39.3/src/transformers/utils/quantization_config.py
                                  # class QuantoConfig(QuantizationConfigMixin):
                                  #     Args:
                                  #        weights (`str`, *optional*, defaults to `"int8"`):
                                  #            The target dtype for the weights after quantization. Supported values are ("float8","int8","int4","int2")
                                  #        activations (`str`, *optional*):
                                  #            The target dtype for the activations after quantization. Supported values are (None,"int8","float8")
                                  #        modules_to_not_convert (`list`, *optional*, default to `None`):
                                  #            The list of modules to not quantize, useful for quantizing models that explicitly require to have
                                  #            some modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers).
                                  #
    GPTQConfig,                   # This is a wrapper class about all possible attributes and features that you can play with a model that has been 
                                  #     loaded using `optimum` api for gptq quantization relying on auto_gptq backend.
                                  # transformers-4.39.3/src/transformers/utils/quantization_config.py
                                  #     class GPTQConfig(QuantizationConfigMixin):
                                  #          Args:
                                  #              bits (`int`):
                                  #                The number of bits to quantize to, supported numbers are (2, 3, 4, 8).
                                  #              tokenizer (`str` or `PreTrainedTokenizerBase`, *optional*):
                                  #                The tokenizer used to process the dataset. You can pass either:
                                  #                  - A custom tokenizer object.
                                  #                  - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                                  #                  - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                                  #                    using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
                                  #              ...
                                  #
    LogitsProcessorList,          # transformers/src/transformers/generation/logits_process.py
                                  #   Create a list of [`LogitsProcessor`] or [`LogitsWarper`] to subsequently process a `scores` input tensor,
                                  #     the 'scores' tensor represents the model's output predictions for each possible next token, based on the input sequence,
                                  #     these scores are used to determine the likelihood of each token being the next in the generated sequence.
    InfNanRemoveLogitsProcessor,  # ...
                                  #
    GenerationConfig,             # transformers/src/transformers/generation/configuration_utils.py
                                  # Class that holds a configuration for a generation task.

    BatchEncoding,                # transformers/src/transformers/tokenization_utils_base.py
                                  #     class BatchEncoding(UserDict): 
                                  #         Args:
                                  #             data (`dict`, *optional*):
                                  #                  Dictionary of lists/arrays/tensors returned by the `__call__`/`encode_plus`/`batch_encode_plus` methods
                                  #                  ('input_ids', 'attention_mask', etc.).
                                  #             encoding (`tokenizers.Encoding` or `Sequence[tokenizers.Encoding]`, *optional*):
                                  #                  If the tokenizer is a fast tokenizer which outputs additional information like mapping from word/character
                                  #                  space to token space the `tokenizers.Encoding` instance or list of instance (for batches) hold this
                                  #                  information.
                                  #             tensor_type (`Union[None, str, TensorType]`, *optional*):
                                  #                  You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
                                  #                      initialization.
                                  #             prepend_batch_axis (`bool`, *optional*, defaults to `False`):
                                  #                 Whether or not to add a batch axis when converting to tensors (see `tensor_type` above).
                                  #             n_sequences (`Optional[int]`, *optional*):
                                  #                 You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
                                  #                 initialization.
                                  #         def to(self, device: Union[str, "torch.device"]) -> "BatchEncoding":
                                  #             Send all values to device by calling `v.to(device)` (PyTorch only).
                                  #     Holds the output of the [`~tokenization_utils_base.PreTrainedTokenizerBase.__call__`]
                                  #     This class is derived from a python dictionary and can be used as a dictionary.
                                  #     
    )

from safetensors import (  # Simple, safe way to store and distribute tensors
        safe_open
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

from enum import Enum, auto

from abc import ABC, abstractmethod

import inspect
import os
import logging
from pathlib import Path


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

    def prompt_to(self, tokenizer: PreTrainedTokenizer, prompt: str) -> Union[List[int], BatchEncoding]:
        """
        Transform a text prompt into a list of token IDs suitable for LLM processing, using the provided tokenizer and current configuration settings.

        This method is intended to prepare user prompts for processing by a large language model (LLM).

        Parameters:
        - tokenizer: An instance of PreTrainedTokenizer, used for converting the text prompt into token IDs.
        - prompt: The text prompt to be transformed.

        Returns:
        - A list of token IDs generated from the prompt, ready for input into an LLM,
          or 
        - BatchEncoding same as `~tokenization_utils_base.PreTrainedTokenizerBase.__call__`
        """
        pass

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



class CausalLMLoRAsInference:
    _logger = None

    @staticmethod
    def logging_setup(log_level = None, log_file: Union[str, Path] = None):
        print(f"#########logging_setup {log_level} {log_file}############")
        valid_log_levels = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
        if log_level not in valid_log_levels and log_level is not None:
            raise ValueError("Invalid log_level. Must be one of logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, or logging.CRITICAL.")
        elif not log_level:
            print(f"logging_setup({log_level}, {log_file}), not set")
            return 

        # create a logger
        logger = logging.getLogger()
        logger.setLevel(log_level)

        # create log format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(funcName)s:%(lineno)d] - %(message)s')

        # create a handler used to output logs to the screen
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        # set format for handler
        console_handler.setFormatter(formatter)
        # add handler to logger
        logger.addHandler(console_handler)

        log2 = "log2console"

        if log_file:
            # create a handler used to write logs to a file
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            log2 = f"log2console and log2file:{log_file}"

        logger.info("logging_setup succ: {log_level} and {log2}")
        CausalLMLoRAsInference._logger = logger


    def __init__(self, model_path, model_name, log_level = None, log_file = None, 
            device_map = "auto", 
            torch_dtype = None, 
            q_bits: int = None, 
            runtime_model_dtype = torch.float32, 
            prefix_seq_len = None, 
            ):
        """
        Args:
            device_map:
                str: auto\cuda\mps\cpu
                dict: { model_module_name1: 0, model_module_name2: 1, ... }

            torch_dtype( (`str` or `torch.dtype` or None): Override the default `torch.dtype` and load the model under a specific `dtype`.
                torch.float16 or torch.bfloat16 or torch.float
                "auto": `torch_dtype` entry in the `config.json` > the first weight of a floating point type in the checkpoint > 
            
            if torch_dtype is not None, then q_bits will be ignored.
            
            q_bits (int): An integer specifying the number of bits used for quantization, must be within the range of 1 to 8.
            
            runtime_model_dtype: Specifies the data type of the model parameters at runtime, especially on CPU,
                Essential for ensuring compatibility across different environments, as some CPUs may 
                  not support certain data types (e.g., float16), potentially leading to runtime errors,
                  for example ==> RuntimeError: "addmm_impl_cpu_" not implemented for 'Half'.
        """
        self.logging_setup(log_level = log_level, log_file = log_file)

        self.model_name = model_name

        if isinstance(device_map, dict):
            self.device_map = device_map
            self.device_type = "cuda"
        else:
            if device_map == "auto":
                self.device_map = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
            else:
                self.device_map = torch.device(device_map)
            self.device_type = self.device_map.type
        logging.info(f"arg device_map is {device_map} ==> {self.device_type}")
        if self.device_type not in ["cpu", "cuda", "mps"]:
            raise ValueError(f"Unsupported device_type: {self.device_type}. Only 'cpu' and 'cuda' and 'mps' are supported.")

        if q_bits is not None and (q_bits < 0 or q_bits > 8):
            raise ValueError(f"Unsupported value for q_bits: {q_bits}. The number of quantization bits (q_bits) must be between 1 and 8, inclusive.")
        self.q_bits = q_bits
        self.quantization_config_list =  [None] * 8
        # 2Bits
        if self.q_bits == 2:
            quanto_config = QuantoConfig(weights="int2")
            self.quantization_config_list[2] = quanto_config
            logging.info("2Bits quantization by Quanto")
        elif self.q_bits == 3:
            gptq_config = GPTQConfig(bits = 3)
            self.quantization_config_list[3] = gptq_config
            logging.info("3Bits quantization by GPTQ")
        # 4Bits
        elif self.q_bits == 4:
            if self.device_type == "cuda":
                bnb_config_nf4 = BitsAndBytesConfig(          # https://huggingface.co/docs/transformers/main_classes/quantization#transformers.BitsAndBytesConfig
                        load_in_4bit=True,                    # enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from bitsandbytes
                        bnb_4bit_quant_type="nf4",            # sets the quantization data type in the bnb.nn.Linear4Bit layers
                        bnb_4bit_use_double_quant=True,       # used for nested quantization where the quantization constants from the first quantization are quantized again
                        bnb_4bit_compute_dtype=torch.bfloat16 # sets the computational type which might be different than the input time
                        )
                self.quantization_config_list[4] = bnb_config_nf4
                logging.info("4Bits quantization by BNB")
            else:
                quanto_config = QuantoConfig(weights="int4")
                self.quantization_config_list[4] = quanto_config
                logging.info("4Bits quantization by Quanto")
        # 8Bits
        elif self.q_bits == 8:
            raise ValueError(f"Unsupported q_bits: {self.q_bits}.")
        else:
            logging.info("no quantization")

        
        # transformers/src/transformers/models/auto/configuration_auto.py
        # This is a generic configuration class that will be instantiated as one of the configuration classes of the library.
        model_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path =model_path, 
                # cache_dir (`str` or `os.PathLike`, *optional*): Path to a directory in which a downloaded pretrained model configuration should be cached.
                # force_download (`bool`, *optional*, defaults to `False`): Whether or not to force the (re-)download the model weights and configuration files.
                # resume_download (`bool`, *optional*, defaults to `False`): Whether or not to delete incompletely received files.
                # proxies (`Dict[str, str]`, *optional*): A dictionary of proxy servers to use by protocol or endpoint.
                # revision (`str`, *optional*, defaults to `"main"`): The specific model version to use.
                # return_unused_kwargs (`bool`, *optional*, defaults to `False`): If `True`, then this functions returns a `Tuple(config, unused_kwargs).
                # trust_remote_code (`bool`, *optional*, defaults to `False`): Whether or not to allow for custom models defined on the Hub in their own modeling files.
                trust_remote_code=True,
                # kwargs(additional keyword arguments, *optional*): 
                )

        # PrefixEncoder
        #
        # The model will define prefix_encoder only if the model config contains pre_seq_len.
        if prefix_seq_len is not None:
            # PrefixEncoder: prompt tuning introduced learnable continuous prompts
            # https://github.com/THUDM/P-tuning-v2/blob/main/model/prefix_encoder.py
            # class PrefixEncoder(torch.nn.Module):
            #     def __init__(self, config):
            #         ...
            #         self.embedding = torch.nn.Embedding(config.pre_seq_len, config.hidden_size)
            #         ...
            model_config.pre_seq_len = prefix_seq_len

        model_from_pretrained_args = {}
        model_from_pretrained_args['device_map'] = self.device_map
        if torch_dtype is not None:
            model_from_pretrained_args['torch_dtype'] = torch_dtype
        else:
            model_from_pretrained_args['quantization_config'] = self.quantization_config_list[self.q_bits] if self.q_bits is not None else None
            logging.debug(f"quantization_config= {model_from_pretrained_args['quantization_config']}")

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
                #
                # config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None -> Configuration for the model to use instead of an automatically loaded configuration.
                config = model_config,
                #
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
                # max_memory = kwargs.pop("max_memory", None)
                # offload_folder = kwargs.pop("offload_folder", None)
                # offload_state_dict = kwargs.pop("offload_state_dict", False)
                # load_in_8bit = kwargs.pop("load_in_8bit", False)
                # load_in_4bit = kwargs.pop("load_in_4bit", False)              -> used to enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from bitsandbytes
                # quantization_config = kwargs.pop("quantization_config", None)
                # subfolder = kwargs.pop("subfolder", "")
                # commit_hash = kwargs.pop("_commit_hash", None)
                # variant = kwargs.pop("variant", None)
                # adapter_kwargs = kwargs.pop("adapter_kwargs", {})
                # adapter_name = kwargs.pop("adapter_name", "default")
                # use_flash_attention_2 = kwargs.pop("use_flash_attention_2", False)
                **model_from_pretrained_args
                )

        self.model_modules_name()


        # Convert the loaded model to float32 to ensure compatibility with CPU operations, avoiding 'Half' data type errors
        #   ==> RuntimeError: "addmm_impl_cpu_" not implemented for 'Half'
        if next(self.model.parameters()).device.type == 'cpu':        
            logging.info(f"self.model.parameters() is being on cpu-ram")
            self.model = self.model.to(dtype = runtime_model_dtype)

        if prefix_seq_len is not None:
            # Check if a specific length for the prefix sequence is provided (`prefix_seq_len` is not None).
            # This also implies that `self.model.transformer.prefix_encoder` exists within our model's architecture,
            # indicating we have a prefix encoder component in the transformer model that can be initialized
            # or manipulated based on the prefix sequence length.
            # By saving the initial state of the `prefix_encoder`, we ensure that we can always revert
            # this component to its original state after any modifications or experiments.
            # This is particularly useful for dynamic adjustments to the prefix encoder's settings or
            # for experimental purposes where the encoder's state needs to be reset to a baseline condition.
            self.initial_prefix_state_dict = self.model.transformer.prefix_encoder.state_dict()


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
        logging.info(f"tokenizer_class_name is:  {self.tokenizer_type_name}")

        import types
        from transformers import PreTrainedTokenizerBase        #
        base_pad_method = getattr(PreTrainedTokenizerBase, '_pad')
        tokenizer_pad_method = getattr(self.tokenizer.__class__, '_pad')
        if tokenizer_pad_method is base_pad_method:
            logging.info(f"Derived classes [{self.tokenizer_type_name}] do not override PreTrainedTokenizerBase._pad")
        else:
            logging.info(f"Derived classes [{self.tokenizer_type_name}] override PreTrainedTokenizerBase._pad")

        self.lora_name_dict = {}
        self.prefix_name_dict = {}

        #transformers/src/transformers/generation/logits_process.py
        self.default_logits_processor = LogitsProcessorList().append(InfNanRemoveLogitsProcessor())


    def model_arch_info(self, model, max_depth=1, depth=0):
        if depth > max_depth:
            return
        for name, child in model.named_children():
            logging.debug('  ' * depth + f"{name} ({child.__class__.__name__})")
            self.model_arch_info(child, max_depth, depth + 1)

    def model_modules_name(self, ):
        for name, module in self.model.named_modules():
            logging.debug(f"name= {name}")

    def load_lora(self, lora_path: str, lora_name: str, model_name: str) -> None:
        """
        Args:
            lora_path (str):
                The local path where the LoRA adapter exists.
            lora_name (`str`):
                The name of the loaded lora adapter, used as the label to select this LoRA adapter.
            model_name (str):
                Use as a key in the loraconfig dictionary to search for the LoraConfig, that a derived class of peft.PeftConfig.
        """
        if lora_name not in self.lora_name_dict:
            self.lora_name_dict[lora_name] = [lora_path, lora_name, model_name]
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
            logging.info(f"Adapter {lora_name}---{lora_path}, loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load adapter {lora_name}---{lora_path}, {e}")
            raise

    def load_prefix(self, prefix_path: str, prefix_name: str, model_name: str) -> None:
        """
        based on:
                 peft/src/peft/tuners/prefix_tuning/model.py
                 https://github.com/THUDM/P-tuning-v2/blob/main/model/prefix_encoder.py

        Loads a model-specific prefix from a specified path. This prefix can be used
        to fine-tune the model on a specific task by adding trainable parameters to 
        the model's input, without altering the pretrained model's internal weights.

        :param prefix_path: str, the file where the prefix data is stored. 
                          
        :param prefix_name: str, the name of the prefix to be loaded. This can be used to
                          select a specific prefix, when used for hot switching.
        :param model_name: str, the name of the model for which the prefix is intended.
                        This ensures that the loaded prefix matches the expected model.
        :return: None. The function performs an action (e.g., loading data, setting state)
              but does not return any value.
        """
        if prefix_name not in self.prefix_name_dict:
            self.prefix_name_dict[prefix_name] = [None, prefix_path, prefix_name, model_name]
        else:
            raise ValueError(f"The prefix_name key '{prefix_name}' already exists in the dictionary.")

        prefix_dict = {}

        try:
            if os.path.exists(os.path.join(prefix_path, "pytorch_model.bin")):  
                prefix_state_dict = torch.load(os.path.join(prefix_path, "pytorch_model.bin"))
                for k, v in prefix_state_dict.items():
                    if k.startswith("transformer.prefix_encoder."):
                        prefix_dict[k[len("transformer.prefix_encoder."):]] = v
                        logging.debug(k[len("transformer.prefix_encoder."):])
                # param_name = next(iter(self.prefix_name_dict[prefix_name][0]))
                # param_device = self.prefix_name_dict[prefix_name][0][param_name].device
                # print(f"The parameter '{param_name}' is stored on {param_device}")
                #self.model.transformer.prefix_encoder.load_state_dict(prefix_dict)
                logging.info(f"Prefix {prefix_name}/pytorch_model.bin, loaded successfully.")
            elif os.path.exists(os.path.join(prefix_path, "model.safetensors")):
                with safe_open(os.path.join(prefix_path, "model.safetensors"), framework="pt") as f:
                    for idx, k in enumerate(f.keys()):
                        key = k[len("transformer.prefix_encoder."):] # remove the prefix "transformer.prefix_encoder." from the string 'k'.
                        prefix_dict[key] = f.get_tensor(k)
                        logging.debug(f"model.safetensors's {idx} key is: {key}")
                logging.info(f"Prefix {prefix_name}/model.safetensors, loaded successfully.")
            else:
                raise FileNotFoundError(f"The required pytorch_model.bin or model.safetensors {prefix_path} does not exist.")
        except Exception as e:
            logging.error(f"Failed to load prefix {prefix_name} in {prefix_path}, {e}")
            raise

        self.prefix_name_dict[prefix_name][0] = prefix_dict


    @torch.inference_mode()
    def chat(self, prompt: str, 
                   lora_chat_tokenids: IChatTokenIDs = None,
                   prefix_chat_tokenids: IChatTokenIDs = None,
                   lora_or_prefix_name: str = None,
                   max_new_tokens: int = 512,
                   logits_processor: LogitsProcessorList  = None,
                   **kwargs
            ) -> str:

        """
        Conducts a chat session using a specified TokenIDs's transformer and configurations.

        Args:
            prompt (str): 

            lora_chat_tokenids: for LoRA
            prefix_chat_tokenids: for PrefixEncoder
                chat_tokenids (IChatTokenIDs): 
                    An instance of IChatTokenIDs used for:
                    prompt_to: transforming prompt into input token IDs suitable for LLM chating.
                    to_completion: transforming generated chat rresponse_tokenids into completion.
        
            lora_or_prefix_name (str): 
                The name of the LoRA Adapter enabled to be used for generating responses.
                 or
                The name of the PrefixEncoder enabled to be used for generating responses.
        
            max_new_tokens (int): 
                The maximum number of tokens to be generated for the response. This limits the length of the model's output.
        
            logits_processor (Optional[LogitsProcessorList]): 
                An optional list of logits processors to apply to the logits before the softmax step, allowing for manipulation of the logits to control the generation process.
        """
        # chat_tokenids selection
        chat_tokenids = None
        if lora_or_prefix_name in self.lora_name_dict:
            self.model.set_adapter( # Sets a specific adapter by forcing the model to use a that adapter and disable the other adapters.
                adapter_name = lora_or_prefix_name # The name of the adapter to set. Can be also a list of strings to set multiple adapters.
                )
            chat_tokenids = lora_chat_tokenids
            logging.info(f"LoRA Adapter {lora_or_prefix_name} enabled.")
        else:
            if self.lora_name_dict:
                # Disable all adapters that are attached to the model. This leads to inferring with the base model only.
                self.model.disable_adapters()
            logging.warn(f"{lora_or_prefix_name} not found in lora_name_dict. For LoRA, falling back to using the base model only.")

        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'prefix_encoder'):
            if lora_or_prefix_name in self.prefix_name_dict:
                # Using .load_state_dict(prefix_dict) updates model parameters with values from prefix_dict,
                # with matching keys dictating which parameters are updated. Subsequent calls with a new state dictionary
                # will overwrite existing parameter values based on these keys.
                #
                # Caution:
                # - Parameters not present in the new state dictionary but exist in the model retain their values,
                #   potentially leading to "stale" data if not intentionally managed.
                # - Keys in the new state dictionary that don't match any parameter name in the model are ignored when strict=False,
                #   or raise an error if strict=True (default behavior), highlighting mismatches between the model and state dictionary.
                try:
                    self.model.transformer.prefix_encoder.load_state_dict(self.prefix_name_dict[lora_or_prefix_name][0])
                except Exception as e:
                    logging.error(f"Exception raise with self.model.transformer.prefix_encoder.load_state_dict( prefix_name_dict[][] ), error message is {e}")
                    raise
                chat_tokenids = prefix_chat_tokenids
                logging.info(f"PrefixEncoder {lora_or_prefix_name} enabled.")
            else:
                try:
                   self.model.transformer.prefix_encoder.load_state_dict(self.initial_prefix_state_dict)
                except Exception as e:
                    logging.error(f"Exception raise with self.model.transformer.prefix_encoder.load_state_dict( initial_prefix_state_dict ), error message is {e}")
                logging.warn(f"PrefixEncoder {lora_or_prefix_name} not found in prefix_name_dict. For PrefixEncoder falling back to using the base model only.")
        else:
            logging.warn("The model does not have transformer and/or prefix_encoder attributes.")



        # generate inputs making
        input_ids = None
        prompt_length = None
        input_config = {}
        device = next(self.model.parameters()).device
        if chat_tokenids and chat_tokenids.prompt_to is not IChatTokenIDs.prompt_to:
            #
            try:
                prompt_inputs = chat_tokenids.prompt_to(self.tokenizer, prompt)
                if isinstance(prompt_inputs, list):
                    input_ids = torch.tensor([prompt_inputs], device = next(self.model.parameters()).device)
                    prompt_length = len(prompt_inputs)
                elif isinstance(prompt_inputs, BatchEncoding):
                    raise NotImplementedError("Handling for BatchEncoding return type is not yet implemented.")
                else:
                    raise TypeError("Unexpected return type from prompt_to method")
            except Exception as e:
                logging.error(f"Exception raise with chat_tokenids.prompt_to(self.tokenizer, prompt), error message is {e}")
                raise
            logging.debug(f"prompt= {prompt_inputs} \ninput_ids= {input_ids}")
        else:
            # the return inputs is a dict like: {'input_ids': tensor([[64790, 64792, 24954]], device='cuda:0'),
            #                                    'attention_mask': tensor([[1, 1, 1]], device='cuda:0'),
            #                                    'position_ids': tensor([[0, 1, 2]], device='cuda:0')
            #                                   }
            inputs = self.tokenizer( #from transformers/src/transformers/tokenization_utils_base.py
                                                 # class PreTrainedTokenizerBase::__call__( ... ) 
                                                 #     Main method to tokenize and prepare for the model one or several sequence(s)
                                                 #        or one or several pair(s) of sequences.
                    # text (`str`, `List[str]`, `List[List[str]]`, *optional*): 
                    text = prompt,               # The sequence or batch of sequences to be encoded.
                    # text_pair (`str`, `List[str]`, `List[List[str]]`, *optional*):
                    # text_target (`str`, `List[str]`, `List[List[str]]`, *optional*):
                    # text_pair_target (`str`, `List[str]`, `List[List[str]]`, *optional*):
                    return_tensors="pt"         #Return PyTorch `torch.Tensor` objects.
                    )
        
            g_inputs = inputs.to(device)
            attention_mask = g_inputs.get('attention_mask', None)
            if attention_mask is not None:
                input_config['attention_mask'] = attention_mask
            position_ids = g_inputs.get('position_ids', None)
            if position_ids is not None:
                input_config['position_ids'] = position_ids
            input_ids = g_inputs['input_ids']
            logging.debug(f"prompt= {prompt} \ng_inputs={g_inputs} \ninput_ids= {input_ids}")


        gen_config_dict = kwargs.copy()
        gen_config_dict['max_new_tokens'] = max_new_tokens
        gen_config_dict['pad_token_id'] = self.tokenizer.pad_token_id
        gen_config_dict['eos_token_id'] = [self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids
        generation_config = GenerationConfig(                      # A large number of these flags control the logits or the stopping criteria of the generation.
                **gen_config_dict                                  # Parameters that control the length of the output
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
                # bos_token_id (`int`, *optional*):               -> The id of the *beginning-of-sequence* token.
                # eos_token_id (`Union[int, List[int]]`, *optional*): -> The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
                                                                  # Generation parameters exclusive to encoder-decoder models
                # encoder_no_repeat_ngram_size (`int`, *optional*, defaults to 0):
                # decoder_start_token_id (`int`, *optional*):
                                                                  # Generation parameters exclusive to [assistant generation](https://arxiv.org/abs/2211.17192)
                # num_assistant_tokens (`int`, *optional*, defaults to 5): 
                # num_assistant_tokens_schedule (`str`, *optional*, defaults to `"heuristic"`):



                # generation_kwargs: Additional generation kwargs will be forwarded to the `generate` function of the model.
                )
        





        try:
            self.model.eval()   # setting evaluation-mode
            with torch.inference_mode(False):   # Temporarily disable inference mode
                generated_tokenids = self.model.generate(
                    input_ids = input_ids,
                    **input_config,
                    generation_config = generation_config,
                    logits_processor = logits_processor if logits_processor is not None else self.default_logits_processor,
                    )
            logging.debug(f"generation_config= {generation_config}")
        except Exception as e:
            logging.error(f"Exception raise with self.model.generate( ... ), error message is {e}")
            raise
       
        # convert to completion text
        completion = None
        if chat_tokenids and chat_tokenids.to_completion is not IChatTokenIDs.to_completion:
            try:
                if prompt_length is not None:
                    response_tokenids = generated_tokenids[:, prompt_length:]
                else:
                    logging.error(f"prompt_length is None, the chat_tokenids.prompt_to() return value may not be a list")
            except Exception as e:
                logging.error(f"Exception raise with response_tokenids = generated_tokenids[:, prompt_length:], error message is {e}")
                raise
            try:
                completion = chat_tokenids.to_completion(self.tokenizer, response_tokenids)
            except Exception as e:
                logging.error(f"Exception raise with chat_tokenids.to_completion(self.tokenizer, response_tokenids), error message is {e}")
                raise
        else:
            try:
                response_tokenids = generated_tokenids[0, input_ids.shape[-1]:]
            except Exception as e:
                logging.error(f"Exception raise with generated_tokenids[0, g_inputs['input_ids'].shape[-1]:], error message is {e}")
                raise
            try:
                completion = self.tokenizer.decode(response_tokenids, skip_special_tokens=True)
            except Exception as e:
                logging.error(f"Exception raise with self.tokenizer.decode(response_tokenids, skip_special_tokens=True), error message is {e}")
                raise

        logging.debug(f"generated_tokenids= {generated_tokenids} \ncompletion= {completion}")

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
