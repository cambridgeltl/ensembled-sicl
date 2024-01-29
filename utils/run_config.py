def create_run_name(args, training_cfg):
    '''
    create unique identical run name for each run
    :param args: from argparse
    :param training_cfg: from configuration file
    :return: unique identical run name
    '''
    run_name = ""

    learning_mode = args.mode
    assert learning_mode in ['ft', 'icl', 'supicl'], (
        ValueError(f"{learning_mode} not supported, should be among ['ft', 'icl', 'supicl']"))
    run_name += learning_mode

    # run_name += f'-template_{args.template}-{args.train_test_split}'
    run_name += f'-trainsize_{args.train_size}'
    if args.imbalance:
        run_name += "-train_imb"
    if args.test_imbalance:
        run_name += "-test_imb"

    if learning_mode in ['ft', 'supicl']:
        train_bz = training_cfg.get('hyper').get('train_batch_size')
        val_bz = training_cfg.get('hyper').get('val_batch_size')
        lr = training_cfg.get('hyper').get('lr')
        run_name = run_name + f"-hyper-train{train_bz}val{val_bz}lr{lr}-"
    else:
        run_name = run_name + "-nohyper-"

    if learning_mode in ['icl', 'supicl']:
        icl_cfg = f"icl-{args.icl_cfg['ic_num']}-retrieve_{'-'.join([k+'_'+v for k, v in args.icl_cfg['retrieve'].items()])}-order_{args.icl_cfg['order']}-"
        run_name = run_name + icl_cfg

    run_name += args.data
    run_name = run_name + '-' + args.task
    if args.input_format is None:
        run_name = run_name + "-" + 'prompt_cycling'
    else:
        run_name = run_name + "-" + str(args.input_format)
    run_name = run_name + "-" + str(args.seed)

    if args.with_logprobs:
        run_name += "-withp"

    if args.mode == 'ft' and args.model_ckpt is None and not args.do_train:
        run_name = 'zeroshot-' + run_name

    if args.model.startswith('meta-llama'):
        run_name = "llama" + run_name
    elif args.model.startswith('t5'):
        run_name = "t5" + run_name

    return run_name