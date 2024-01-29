def create_peft_config(config):
    from peft import (
        LoraConfig,
        TaskType
    )
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=config['inference_mode'],
        r=int(config['r']),
        lora_alpha=int(config['lora_alpha']),
        lora_dropout=float(config['lora_dropout']),
        target_modules=config['target_modules']
    )
    return peft_config