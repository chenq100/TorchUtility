# cython: language_level=3
### Reference: https://huggingface.co/docs/transformers/main/peft

from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,            # transformers-4.39.3/src/transformers/trainer.py
                            # Trainer is a simple but feature-complete training and eval loop for PyTorch, optimized for 🤗 Transformers.
        BitsAndBytesConfig,
        QuantoConfig,       # class QuantoConfig(QuantizationConfigMixin): Reference: transformers-4.39.3/src/transformers/utils/quantization_config.py
                            #     This is a wrapper class about all possible attributes and features that you can play with a model 
                            #     that has been loaded using `quanto`.
        GPTQConfig,         # class GPTQConfig(QuantizationConfigMixin): Reference: transformers-4.39.3/src/transformers/utils/quantization_config.py
                            #     This is a wrapper class about all possible attributes and features that you can play with a model 
                            #     that has been loaded using `optimum` api for gptq quantization relying on auto_gptq backend.
        TrainingArguments,  # transformers-4.39.3/src/transformers/training_args.py
                            # class TrainingArguments:
        )

from peft import (
        LoraConfig,         # class LoraConfig(PeftConfig): Reference: peft-0.10.0/src/peft/tuners/lora/config.py
        LoftQConfig,        # class LoftQConfig: Reference: peft-0.10.0/src/peft/tuners/lora/config.py
        TaskType
        )

from datasets import (
        Dataset,            # datasets-2.18.0/src/datasets/arrow_dataset.py
                            # class Dataset(DatasetInfoMixin, IndexableMixin, TensorflowDatasetMixin):
                            #     """A Dataset backed by an Arrow table."""
                            #     def from_dict( ... )
                            #         """Convert `dict` to a `pyarrow.Table` to create a [`Dataset`]."""
                            #
        load_dataset,       # datasets-2.18.0/src/datasets/load.py
                            # def load_dataset( ... )
                            #     """
                            #         load a dataset from the Hugging Face Hub, or a local dataset,
                            #         can find the list of datasets on the [Hub](https://huggingface.co/datasets)
                            #     """
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
    
    def __init__(self, model_path, model_name, log_level = None, log_file = None, q_bits: int = None, device_map = "auto", runtime_model_dtype = torch.float32, prefix_seq_len = None):
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
            if torch.backends.mps.is_available():
                self.device_map = torch.device("mps")
            elif torch.cuda.is_available():
                self.device_map = torch.device("cuda:0")
            else:
                self.device_map = torch.device("cpu")
        elif isinstance(device_map, dict):
            processed_device_map = {}
            for key, value in device_map.items():
                if isinstance(value, int):
                    processed_device_map[key] = torch.device(f"cuda:{value}")
                elif isinstance(value, str) and value.startswith("cuda:"):
                    processed_device_map[key] = torch.device(value)
                else:
                    raise ValueError(f"Invalid device specification {value}. Must be an integer or a 'cuda:x' string.")
            self.device_map = processed_device_map
        else:
            self.device_map = torch.device(device_map)
        logging.info(f"arg device_map is {device_map} ==> {self.device_map}")
        if self.device_map.type not in ["cpu", "cuda", "mps"]:
            raise ValueError(f"Unsupported device_map: {self.device_map}. Only 'cpu' and 'cuda' and 'mps' are supported.")

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
            if self.device_map.type == "cuda":
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


    def add_lora(self, lora_path: str, lora_name: str, model_name: str, 
        r: int = 8, 
        target_modules: Optional[Union[list[str], str]] = "all-linear",
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
            self.lora_name_dict[lora_name] = [lora_path, lora_name, model_name]
        else:
            raise ValueError(f"The lora_name key '{lora_name}' already exists in the dictionary.")

        lora_config_args = {}
        if modules_to_save is not None:
            lora_config_args['modules_to_save'] = modules_to_save
        if layers_to_transform is not None:
            lora_config_args['layers_to_transform'] = layers_to_transform
        if layers_pattern is not None:
            lora_config_args['layers_pattern'] = layers_pattern
        if rank_pattern is not None:
            lora_config_args['rank_pattern'] = rank_pattern
        if alpha_pattern is not None:
            lora_config_args['alpha_pattern'] = alpha_pattern
        if megatron_config is not None:
            lora_config_args['megatron_config'] = megatron_config
        if megatron_core is not None:
            lora_config_args['megatron_core'] = megatron_core
        if layer_replication is not None:
            lora_config_args['layer_replication'] = layer_replication


        lora_config = LoraConfig(
                r = r,
                target_modules = target_modules,           # The names of the modules to apply the adapter to.
                lora_alpha = lora_alpha,                   # The alpha parameter for Lora scaling.
                lora_dropout = lora_dropout,               # The dropout probability for Lora layers.
                fan_in_fan_out = fan_in_fan_out,           # Set this to True if the layer to replace stores weight like (fan_in, fan_out).
                bias = bias,                               # Bias type for LoRA: Can be 'none', 'all' or 'lora_only'. the corresponding biases will be updated during training.
                use_rslora = use_rslora,                   # Sets the adapter scaling factor to `lora_alpha/math.sqrt(r)`
                #modules_to_save = modules_to_save,         # List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
                init_lora_weights = init_lora_weights,     # How to initialize the weights of the adapter layers.
                #layers_to_transform = layers_to_transform, # The layer indices to transform, means that these layers will be embedded in the LoRA matrix.
                #layers_pattern = layers_pattern,           # The layer pattern name, used to further refine or specify the layers selected from layers_to_transform ?
                #rank_pattern = rank_pattern,               # {"layer name": specific r-value} as opposed to uniform value: 'r'.
                #alpha_pattern = alpha_pattern,             # {"layer name": specific alpha-value} as opposed to uniform value: 'lora_alpha'.
                #megatron_config = megatron_config,         # The TransformerConfig arguments for Megatron(Nvidia), used to create LoRA's parallel linear layer.
                #megatron_core = megatron_core,             # The core module from Megatron to use, defaults to `"megatron.core"`.
                loftq_config = loftq_config,               # The configuration of LoftQ, will be used to quantize the backbone weights and initialize Lora layers.
                                                           #     LoftQ introduces a novel quantization framework tailored for LoRA fine-tuning,
                                                           #     effectively bridging the gap between quantized and full-precision models by finding an optimal low-rank initialization,
                                                           #     thereby significantly enhancing model generalization and performance on downstream tasks.
                use_dora = use_dora,                       # Enable 'Weight-Decomposed Low-Rank Adaptation' (DoRA).
                #layer_replication = layer_replication      #  Build a new stack of layers by stacking the original model layers according to the ranges specified.
                **lora_config_args
                )

        self.lora_name_dict[lora_name].append(lora_config)

        self.model.add_adapter(lora_config, adapter_name = lora_name)

    def train(self, 
            lora_name: str, 
            samples: Union[List[Dict[str, str]], str, Path], 
            max_length,
            training_args: TrainingArguments,
            ) -> None:

        if samples and isinstance(samples, list):
            input_ids_list = []
            target_ids_list = []
            attention_mask_list = []
            for sample in samples:
                tokenized_ids = self.tokenizer.apply_chat_template(  # transformers-4.39.3/src/transformers/tokenization_utils_base.py
                                                                 # -> Union[str, List[int]], A list of token ids representing the tokenized chat so far, including control tokens.
                        # conversation (Union[List[Dict[str, str]], "Conversation"]): A Conversation object or list of dicts with "role" and "content" keys, representing the chat history so far.
                        conversation=[{"role": sample.get("role", "user"), "content": sample.get("content", "")}],    # [{"role": ..., "content": ...}]
                        # chat_template (str, *optional*): A Jinja template to use for this conversion.
                        chat_template=None,
                        # add_generation_prompt (bool, *optional*): Whether to end the prompt with the token(s) that indicate the start of an assistant message.
                        add_generation_prompt=False,
                        # tokenize (`bool`, defaults to `True`): Whether to tokenize the output. If `False`, the output will be a string.
                        tokenize=True,
                        # padding (`bool`, defaults to `False`): Whether to pad sequences to the maximum length. Has no effect if tokenize is `False`.
                        padding=True,
                        # truncation (`bool`, defaults to `False`): Whether to truncate sequences at the maximum length. Has no effect if tokenize is `False`.
                        truncation=True,
                        # max_length (`int`, *optional*): Maximum length (in tokens) to use for padding or truncation.
                        max_length=max_length,
                        # return_tensors (`str` or [`~utils.TensorType`], *optional*): If set, will return tensors of a particular framework. - `'pt'`: Return PyTorch `torch.Tensor` objects.
                        return_tensors="pt"
                        )
                tokenized_ids_tensor = torch.tensor(tokenized_ids, dtype=torch.long)
                logging.debug(f"self.tokenizer.apply_chat_template got {tokenized_ids_tensor}")
                input_ids_list.append(tokenized_ids_tensor)
                attention_mask = torch.tensor([1 if id != self.tokenizer.pad_token_id else 0 for id in tokenized_ids], dtype=torch.long)
                attention_mask_list.append(attention_mask)
                target_ids_list.append(tokenized_ids_tensor.clone())

            input_ids_tensor = torch.stack(input_ids_list)
            attention_mask_tensor = torch.stack(attention_mask_list)
            target_ids_tensor = torch.stack(target_ids_list)
            sample_dict = {
                    'input_ids': input_ids_tensor,
                    'attention_mask': attention_mask_tensor,
                    'target_ids': target_ids_tensor
                }
            train_dataset = Dataset.from_dict(sample_dict)
            logging.info(f"creating train_dataset by samples")
            logging.debug(samples)
        elif isinstance(samples, (str, Path)):
            train_dataset = load_dataset(
                            # path: str,
                            # name: Optional[str] = None,
                            # data_dir: Optional[str] = None,
                            # data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
                            # split: Optional[Union[str, Split]] = None,
                            # cache_dir: Optional[str] = None,
                            # features: Optional[Features] = None,
                            # download_config: Optional[DownloadConfig] = None,
                            # download_mode: Optional[Union[DownloadMode, str]] = None,
                            # verification_mode: Optional[Union[VerificationMode, str]] = None,
                            # ignore_verifications="deprecated",
                            # keep_in_memory: Optional[bool] = None,
                            # save_infos: bool = False,
                            # revision: Optional[Union[str, Version]] = None,
                            # token: Optional[Union[bool, str]] = None,
                            # use_auth_token="deprecated",
                            # task="deprecated",
                            # streaming: bool = False,
                            # num_proc: Optional[int] = None,
                            # storage_options: Optional[Dict] = None,
                            # trust_remote_code: bool = None,
                            # **config_kwargs,
                            )
            logging.info(f"creating train_dataset by load_dataset( {samples} )")
        else:
            raise TypeError("Unsupported input type. Expected a list for custom conversion for from_dict or a valid identifier for load_dataset!")
        logging.debug(train_dataset)



        device = next(self.model.parameters()).device
        logging.debug(f"when training self.model is on device: {device} - {self.model.hf_device_map.values()}")
        self.model.train()
        trainer = Trainer(
                # model: Union[PreTrainedModel, nn.Module] = None, 
                model=self.model, 
                # args: TrainingArguments = None,
                args = training_args,
                # data_collator: Optional[DataCollator] = None,
                # train_dataset: Optional[Dataset] = None,
                train_dataset=train_dataset, 
                # eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
                # tokenizer: Optional[PreTrainedTokenizerBase] = None,
                tokenizer=self.tokenizer,
                # model_init: Optional[Callable[[], PreTrainedModel]] = None,
                # compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
                # callbacks: Optional[List[TrainerCallback]] = None,
                # optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                # preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                )

        logging.info(f"Trainer start training...")
        trainer.train()











if __name__ == "__main__":
    sample_list = [
                    {"instruction": "Do something", "input": "Data 1", "output": "Result 1"},
                    {"instruction": "Do another thing", "input": "Data 2", "output": "Result 2"},
                    {"instruction": "Do a different thing", "input": "Data 3", "output": "Result 3"},
                    {"instruction": "Repeat something", "input": "Data 4", "output": "Result 4"},
                  ]
    CausalLMLoRAsTrain.logging_setup(log_level = logging.DEBUG)
    CausalLMLoRAsTrain.training(lora_name = "test", samples = sample_list, training_args = None)
