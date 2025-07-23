from src.train_load_task import *
from src.roberta_model_task import *
from src.evaluation_task import all_metrics
from src.args_parse import *
from src.vocab_all import *
from src.process import *
from torch.utils.data.dataloader import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import AdamW, get_scheduler,AutoConfig
from transformers import AutoTokenizer,RobertaModel
from tqdm.autonotebook import tqdm
import random
import logging
import math
import torch
import datetime
from src.utils import compute_cosine_similarity_loss, compute_class_freq_and_train_num,head_tail_mask_bycoverage, update_head_tail_mask

logger = logging.getLogger(__name__)
set_random_seed(random_seed=42)


def main():
    args = create_args_parser()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    
    lab_file = args.label_titlefile
    with open(lab_file, 'rb') as file:
        labDescVec = pickle.load(file)
        labDescVec = labDescVec['id2descVec']
    labDescVec = torch.nn.Parameter(torch.tensor(labDescVec, dtype=torch.float32))

    raw_datasets = load_data(args.data_dir)
    print(raw_datasets)
    vocab = Vocab(args, raw_datasets)
    label_to_id = vocab.label_to_id
    num_labels = len(label_to_id)

    print(len(vocab.label_dict['train']))
    print(len(vocab.label_dict['valid']))
    print(len(vocab.label_dict['test']))
    remove_columns = raw_datasets["train"].column_names
    print(remove_columns)

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,do_lower_case=True)

    if "Text" in remove_columns:
        text_name="Text"
    elif "text" in remove_columns:
        text_name = "text"
    if "Full_Labels" in remove_columns:
        label_columns="Full_Labels"
    elif "target" in remove_columns:
        label_columns = "target"


    def getitem(examples):
        label_list=[]
        texts = ((examples[text_name],))
        result = tokenizer(*texts, padding=False, max_length=args.max_seq_length, truncation=True,add_special_tokens=True)
        if "Full_Labels"==label_columns:
            for labels in examples["Full_Labels"]:
                label_list.append([vocab.label_to_id[label.strip()] for label in labels.strip().split('|')])
        elif "target"==label_columns:
            for labels in examples["target"]:
                label_list.append([vocab.label_to_id[label.strip()] for label in labels.strip().split(',')])

        result["label_ids"] = label_list
        return result

    def data_collator_train(features):
        batch = dict()
        if args.dataEnhance:
            for i in range(len(features)):
                len_fea = int(len(features[i]['input_ids'][1:-1]))
                if random.random() < args.dataEnhanceRatio / 2:
                    features[i]['input_ids'][1:-1] = torch.tensor(features[i]['input_ids'])[1:-1][np.random.permutation(len_fea)].tolist()
                if random.random() < args.dataEnhanceRatio:
                    features[i]['input_ids'][1:-1] = torch.tensor(features[i]['input_ids'])[1:-1][range(len_fea)[::-1]].tolist()
        max_length = max([len(f["input_ids"]) for f in features])
        if max_length % args.chunk_size != 0:
            max_length = max_length - (max_length % args.chunk_size) + args.chunk_size

        batch["input_ids"] = torch.tensor([
            f["input_ids"] + [tokenizer.pad_token_id] * (max_length - len(f["input_ids"]))
            for f in features
        ]).contiguous().view((len(features), -1, args.chunk_size))

        label_ids = torch.zeros((len(features),num_labels))
        for i, f in enumerate(features):
            for label in f["label_ids"]:
                label_ids[i, label] = 1
        batch["label_ids"] = label_ids

        return batch

    def data_collator(features):
        batch = dict()
        max_length = max([len(f["input_ids"]) for f in features])
        if max_length % args.chunk_size != 0:
            max_length = max_length - (max_length % args.chunk_size) + args.chunk_size
        batch["input_ids"] = torch.tensor([
            f["input_ids"] + [tokenizer.pad_token_id] * (max_length - len(f["input_ids"]))
            for f in features
        ]).contiguous().view((len(features), -1, args.chunk_size))
        label_ids = torch.zeros((len(features),num_labels))

        for i, f in enumerate(features):
            for label in f["label_ids"]:
                label_ids[i, label] = 1
        batch["label_ids"] = label_ids
        return batch
    processed_datasets = raw_datasets.map(getitem, batched=True,
                                          remove_columns=remove_columns)
    print(processed_datasets)
    
    class_freq = compute_class_freq_and_train_num(label_to_id, processed_datasets["train"], freq_cutoff=0)
    head_tail_mask, tail_labels = head_tail_mask_bycoverage(class_freq,label_to_id,0.5)
    
    print("=========================================")

    train_dataloader = DataLoader(processed_datasets["train"], shuffle=True, collate_fn=data_collator_train,
                              batch_size=args.batch_size, pin_memory=True)
    eval_dataloader = DataLoader(processed_datasets["valid"], collate_fn=data_collator, batch_size=args.batch_size, pin_memory=True)
    test_dataloader = DataLoader(processed_datasets["test"], collate_fn=data_collator, batch_size=args.batch_size, pin_memory=True)
    
    if "roberta"==args.model_type:
        model = Roberta_model.from_pretrained(args.model_name_or_path,config=config,args=args,vocab=vocab)


    if args.optimiser.lower() == "adamw":
        if args.use_different_lr:
            ignored_params = list(map(id, model.roberta.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
            optimizer_grouped_parameters = [
                {
                    "params": model.roberta.parameters(),
                    "lr": args.plm_lr,
                },
                {
                    "params": base_params,
                    "lr": args.lr,
                },
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        else:
            betas = (0.9, 0.999)
            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=args.lr, betas=betas, weight_decay=args.weight_decay)
            print("optimizer", optimizer)

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    eval_dataloader = accelerator.prepare(eval_dataloader)
    test_dataloader = accelerator.prepare(test_dataloader)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.n_epoch * num_update_steps_per_epoch

    T_epochs=args.num_train_epochs

    if args.use_lr_scheduler:
        itersPerEpoch = num_update_steps_per_epoch
        print("itersPerEpoch", itersPerEpoch)
        epoch = T_epochs
        warmupEpochs = args.warmup
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmupEpochs * itersPerEpoch,
                                                       num_training_steps=epoch * itersPerEpoch)

    criterions = nn.BCEWithLogitsLoss()
    criterions_tail = nn.BCEWithLogitsLoss()

    print("optimizer", optimizer)
    print("lr_scheduler", lr_scheduler)

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    batch_id = 0
    metrics_max = None
    metrics_max_val=-1
    epoch_max=0
    epoch_max_test=0
    lambda_tail = 1.0 / np.sqrt(sum(head_tail_mask) + 1e-6)

    for epoch in tqdm(range(args.n_epoch)):
        lambda_tail = lambda_tail * 1.5
        print(" ")
        print(datetime.datetime.now().strftime('%H:%M:%S'))
        print("第%d轮" % (epoch + 1))
        model.train()
        optimizer.zero_grad()
        losses = []
        epoch_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            batch_id += 1
            outputs = model(**batch)
            dynamic_embedding,logits = outputs['dynamic_embedding'], outputs['logits']
            
            head_tail_mask_tensor = torch.tensor(head_tail_mask, device=logits.device)
            logits_head = logits[:, head_tail_mask_tensor == 1]
            logits_tail = logits[:, head_tail_mask_tensor == 0]
            label_ids_head = batch['label_ids'][:, head_tail_mask_tensor == 1]
            label_ids_tail = batch['label_ids'][:, head_tail_mask_tensor == 0]
            loss_tail = criterions_tail(logits_tail, label_ids_tail.float())

            main_loss = criterions(logits, batch['label_ids'])
            similirity_loss = compute_cosine_similarity_loss(dynamic_embedding,labDescVec)

            loss = main_loss + loss_tail * 0.05 + similirity_loss * 0.1
            loss = loss / args.gradient_accumulation_steps

            accelerator.backward(loss)
            losses.append(loss.item())
            epoch_loss += loss.item()
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                progress_bar.set_postfix(loss=epoch_loss / batch_id)

        if epoch % 3 == 0:
            head_tail_mask = update_head_tail_mask(head_tail_mask, class_freq, tail_labels, top_percentage=0.2)
            
        model.eval()
        all_preds = []
        all_preds_raw = []
        all_labels = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
                logits = outputs['logits']
            preds_raw = logits.sigmoid().cpu()
            preds = (preds_raw > 0.5).int()
            all_preds_raw.extend(list(preds_raw))
            all_preds.extend(list(preds))
            all_labels.extend(list(batch["label_ids"].cpu().numpy()))
        all_preds_raw = np.stack(all_preds_raw)
        all_preds = np.stack(all_preds)
        all_labels = np.stack(all_labels)
        metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw)
        print("学习率lr{}:".format(optimizer.param_groups[0]["lr"]),"学习率lr{}:".format(optimizer.param_groups[1]["lr"]), "loss: ", np.mean(losses).item())
        print(f"验证集: F1_micro: {metrics['f1_micro']:.4f}, F1_macro: {metrics['f1_macro']:.4f}, auc_micro: {metrics['auc_micro']:.4f}, auc_macro: {metrics['auc_macro']:.4f}, prec_at_8: {metrics['prec_at_8']:.4f}")

        metrics_max_2,best_thre_2 = ans_test(test_dataloader, model)

        if metrics_max is None:
            metrics_max = metrics_max_2
            best_thre = best_thre_2
        else:
            if metrics_max['f1_macro'] < metrics_max_2['f1_macro']:
                epoch_max_test=epoch+1
                metrics_max = metrics_max_2
                best_thre = best_thre_2

        if metrics_max_val<metrics['f1_macro']:
            epoch_max=epoch+1
            metrics_max_val = metrics['f1_macro']
            if args.best_model_path is not None:
                if args.best_model_path is not None:
                    os.makedirs(args.best_model_path, exist_ok=True)
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(args.best_model_path, save_function=accelerator.save)
                checkpoint = {'epoch': epoch + 1,
                              'metrics':metrics_max,
                              'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict()}
                torch.save(checkpoint, args.best_model_path + "check.pth")
                print("更新了更好的模型参数")

    print("验证集最好的epoch:",epoch_max)
    print(f"测试集最好的epoch: {epoch_max_test}, 最好的threshould:{best_thre:.2f}, ",",".join(f"{k}: {v:.4f}" for k, v in metrics_max.items()))


if __name__ == "__main__":
    main()