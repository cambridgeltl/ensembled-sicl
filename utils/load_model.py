import torch

def load_model(args):
    '''
    return the tokenizer and model
    :param args: from argparse
    :return: tuple(model, tokenizer)
    '''
    model_type = args.model
    if model_type.startswith('google/flan-t5') or model_type.startswith('t5'):
        model_size = "large" if args.model_version is None else args.model_version
        model_name = model_type + '-' + model_size
    elif model_type.lower().startswith('meta-llama'):
        model_name = model_type
    else:
        raise ValueError(f'Unsupported model type: {model_type}. ')

    tokenizer = load_tokenizer(model_name)
    model = load_huggingface_model(args, model_name, tokenizer)

    if model_type.lower().startswith('meta-llama'):
        assert args.peft_config is not None, ValueError("PEFT config is not initialized in cfg file! ")
        from peft import (
            get_peft_model,
            prepare_model_for_int8_training,
        )

        peft_config = args.peft_config

        # prepare int-8 model for training
        model = prepare_model_for_int8_training(model)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    if model_type.lower().startswith('google') and args.model_version == 'xl':
        assert args.peft_config is not None, ValueError("PEFT config is not initialized in cfg file! ")
        from peft import (
            get_peft_model,
            prepare_model_for_int8_training,
        )

        peft_config = args.peft_config

        # prepare int-8 model for training
        model = prepare_model_for_int8_training(model)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, tokenizer


def load_tokenizer(model_name: str):
    if model_name.startswith('google/flan-t5') or model_name.startswith('t5'):
        from transformers import T5Tokenizer
        return T5Tokenizer.from_pretrained(model_name)
    elif model_name.lower().startswith('meta-llama'):
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            model_name,
            token='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',
            # add_eos_token=True
        )
        # tokenizer.pad_token_id = 18610
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        return tokenizer
    else:
        raise ValueError(f"{model_name} not supported. modify the load_model function in utils/load_model.py")


def load_huggingface_model(args, model_name, tokenizer):
    if model_name.startswith('google/flan-t5') or model_name.startswith('t5'):
        from transformers import T5ForConditionalGeneration
        return T5ForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=args.cache_dir
        )
    elif model_name.lower().startswith('meta-llama'):
        from transformers import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map='auto',
            torch_dtype=torch.float16,
            token='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',
            cache_dir=args.cache_dir
        )
        # model.resize_token_embeddings(len(tokenizer))
        return model
    else:
        raise ValueError(f"{model_name} not supported. modify the load_model function in utils/load_model.py")