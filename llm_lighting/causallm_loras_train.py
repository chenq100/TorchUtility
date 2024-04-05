# cython: language_level=3
### Reference: https://huggingface.co/docs/transformers/main/peft

from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        )

from peft import (
        LoraConfig,         # class LoraConfig(PeftConfig): Reference: peft/src/peft/tuners/lora/config.py
        TaskType
        )


class CausalLMLoRAsTrain:
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
    
    def __init__(self, model_path, model_name, log_level = None, log_file = None, q_bits: int = None, device_map = "auto",):
        """
        Args:
            q_bits (int): An integer specifying the number of bits used for quantization, must be within the range of 1 to 8.
            runtime_model_dtype: Specifies the data type of the model parameters at runtime, especially on CPU,
                Essential for ensuring compatibility across different environments, as some CPUs may
                  not support certain data types (e.g., float16), potentially leading to runtime errors,
                  for example ==> RuntimeError: "addmm_impl_cpu_" not implemented for 'Half'.
        """
        self.logging_setup(log_level = log_level, log_file = log_file)

        self.model_name = model_name

        if q_bits is not None and (q_bits < 0 or q_bits > 8):
            raise ValueError(f"Unsupported value for q_bits: {q_bits}. The number of quantization bits (q_bits) must be between 1 and 8, inclusive.")
        self.q_bits = q_bits

        if device_map == "auto":
            self.device_map = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device_map = torch.device(device_map)
        logging.info(f"arg device_map is {device_map} ==> {self.device_map}")
        if self.device_map.type not in ["cpu", "cuda"]:
            raise ValueError(f"Unsupported device_map: {self.device_map}. Only 'cpu' and 'cuda' are supported.")

        self.quantization_config_list =  [None] * 8
        # 2Bits
        if self.q_bits == 2:
            quanto_config = QuantoConfig(weights="int2")
            self.quantization_config_list[2] = quanto_config
            logging.info("2Bits quantization")
        # 4Bits
        elif self.q_bits == 4:
            if self.device_map.type == "cuda":
                bnb_config_nf4 = BitsAndBytesConfig(          # https://huggingface.co/docs/transformers/main_classes/quantization#transformers.BitsAndBytesConfig
                        load_in_4bit=True,                    # enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from bitsandbytes
                        bnb_4bit_quant_type="nf4",            # sets the quantization data type in the bnb.nn.Linear4Bit layers
                        bnb_4bit_use_double_quant=True,       # used for nested quantization where the quantization constants from the first quantization are quantized again
                        bnb_4bit_compute_dtype=torch.bfloat16 # sets the computational type which might be different than the input time
                        )
                self.quantization_config_list[4] = bnb_config_nf4
                logging.info("4Bits quantization for cuda")
            else:
                quanto_config = QuantoConfig(weights="int4")
                self.quantization_config_list[4] = quanto_config
                logging.info("4Bits quantization")
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
                device_map = self.device_map,
                # max_memory = kwargs.pop("max_memory", None)
                # offload_folder = kwargs.pop("offload_folder", None)
                # offload_state_dict = kwargs.pop("offload_state_dict", False)
                # load_in_8bit = kwargs.pop("load_in_8bit", False)
                # load_in_4bit = kwargs.pop("load_in_4bit", False)              -> used to enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from bitsandbytes
                # quantization_config = kwargs.pop("quantization_config", None)
                quantization_config = self.quantization_config_list[self.q_bits] if self.q_bits is not None else None,
                # subfolder = kwargs.pop("subfolder", "")
                # commit_hash = kwargs.pop("_commit_hash", None)
                # variant = kwargs.pop("variant", None)
                # adapter_kwargs = kwargs.pop("adapter_kwargs", {})
                # adapter_name = kwargs.pop("adapter_name", "default")
                # use_flash_attention_2 = kwargs.pop("use_flash_attention_2", False)
                )

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


    def add_lora(self, lora_path: str, lora_name: str, model_name: str, 
        r: int = 8, 
        target_modules: Optional[Union[list[str], str]] = None,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        bias: Literal["none", "all", "lora_only"] = "none",
        use_rslora: bool = False,
        modules_to_save: Optional[list[str]] = None,
        init_lora_weights: bool | Literal["gaussian", "loftq"] = True,
        layers_to_transform: Optional[Union[list[int], int]] = None,
        layers_pattern: Optional[Union[list[str], str]] = None,
        rank_pattern:  Optional[dict] = None, 
        alpha_pattern: Optional[dict] = None,
        megatron_config: Optional[dict] = None,
        megatron_core: Optional[str] = None,
        loftq_config: Union[LoftQConfig, dict] = dict,
        use_dora: bool = False,
        layer_replication: Optional[list[tuple[int, int]]] = None,
        ) -> None:
        """
        Args:
            lora_path (str):
                The local path where the LoRA adapter save to.
            lora_name (`str`):
                The name of the added lora adapter, used as the label to select this LoRA adapter.
            model_name (str):
                Use as a key in the loraconfig dictionary to search for the LoraConfig, that a derived class of peft.PeftConfig.
        """
        if lora_name not in self.lora_name_dict:
            self.lora_name_dict[lora_name] = [lora_config, lora_path, lora_name, model_name]
        else:
            raise ValueError(f"The lora_name key '{lora_name}' already exists in the dictionary.")

        lora_config = LoraConfig(
                r = r,
                target_modules = target_modules,           # The names of the modules to apply the adapter to.
                lora_alpha = lora_alpha,                   # The alpha parameter for Lora scaling.
                lora_dropout = lora_dropout,               # The dropout probability for Lora layers.
                fan_in_fan_out = fan_in_fan_out,           # Set this to True if the layer to replace stores weight like (fan_in, fan_out).
                bias = bias,                               # Bias type for LoRA: Can be 'none', 'all' or 'lora_only'. the corresponding biases will be updated during training.
                use_rslora = use_rslora,                   # Sets the adapter scaling factor to `lora_alpha/math.sqrt(r)`
                modules_to_save = modules_to_save,         # List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
                init_lora_weights = init_lora_weights,     # How to initialize the weights of the adapter layers.
                layers_to_transform = layers_to_transform, # The layer indices to transform, means that these layers will be embedded in the LoRA matrix.
                layers_pattern = layers_pattern,           # The layer pattern name, used to further refine or specify the layers selected from layers_to_transform ?
                rank_pattern = rank_pattern,               # {"layer name": specific r-value} as opposed to uniform value: 'r'.
                alpha_pattern = alpha_pattern,             # {"layer name": specific alpha-value} as opposed to uniform value: 'lora_alpha'.
                megatron_config = megatron_config,         # The TransformerConfig arguments for Megatron(Nvidia), used to create LoRA's parallel linear layer.
                megatron_core = megatron_core,             # The core module from Megatron to use, defaults to `"megatron.core"`.
                loftq_config = loftq_config,               # The configuration of LoftQ, will be used to quantize the backbone weights and initialize Lora layers.
                                                           #     LoftQ introduces a novel quantization framework tailored for LoRA fine-tuning,
                                                           #     effectively bridging the gap between quantized and full-precision models by finding an optimal low-rank initialization,
                                                           #     thereby significantly enhancing model generalization and performance on downstream tasks.
                use_dora = use_dora,                       # Enable 'Weight-Decomposed Low-Rank Adaptation' (DoRA).
                layer_replication = layer_replication      #  Build a new stack of layers by stacking the original model layers according to the ranges specified.
                )


        self.model.add_adapter(adapter_name = lora_name, lora_config = lora_config)

    def training(self, lora_name: str) -> None:











