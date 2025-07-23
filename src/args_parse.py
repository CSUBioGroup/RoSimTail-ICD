import argparse

def create_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--add_loss_tail", type=int, default="0", help="是否添加额外的尾部标签损失")
    parser.add_argument("--trans_drop", type=float, default=0.1, help="drop")
    parser.add_argument("--plm_lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--is_padding", type=int, default=0, help="token影藏参数")
    parser.add_argument("--num_experts",type=int,default=4,help="专家个数")
    parser.add_argument("--topk", type=int, default=1, help="专家topk")
    parser.add_argument("--expert_dk", type=int, default=64, help="专家隐藏层")
    parser.add_argument('--model_name_or_path', type=str, default="models/RoBERTa-base-PM-M3-Voc-distill-align-hf/")
    parser.add_argument("--model_type",type=str,default="roberta",help="预训练模型类型",
                        choices=["bert", "roberta", "longformer"])
    parser.add_argument("--chunk_size",type=int,default=128,help="块大小")
    parser.add_argument("--best_model_path", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--use_different_lr",action="store_true", help="是否使用不同学习率")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="执行向后/更新过程之前要累积的更新步骤数",
                    )
    parser.add_argument('--data_dir', type=str, default="../data/mimicdata/mimiciii_clean/")
    parser.add_argument('--code_50', type=int, choices=[0, 1], default=0, help="是否是50的")
    parser.add_argument('--label_titlefile', type=str,
                        default="../data/mimicdata/mimiciii_clean/split_lable_title_skipgram_full_1024.pkl")
    parser.add_argument('--train_title_emb', action="store_true", help="是否进行训练标签的emb")
    parser.add_argument("--dataEnhance", action="store_true", help="是否使用数据增强")
    parser.add_argument("--dataEnhanceRatio", type=float, default=0.2, help="数据增强率")
    parser.add_argument("--is_trans", action="store_true", help="是否使用Transformer")
    parser.add_argument("--multiNum", type=int, default=4, help="Transformer的头数")
    parser.add_argument("--dk", type=int, default=128, help="Transformer的影藏参数")
    parser.add_argument("--lr_cosine", action="store_true", help="是否使用cosine")
    parser.add_argument("--warmup", type=int, default=2, help="预热")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Total number of training epochs to perform.")
    parser.add_argument("--hidden_size", type=int, default=5, help="The size of the hidden layer")
    parser.add_argument('--iter_layer', type=int, default=1, help="迭代的层数")
    parser.add_argument('--reshape_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument("--n_epoch", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5, help="Early Stopping")
    parser.add_argument("--optimiser", type=str, choices=["adagrad", "adam", "sgd", "adadelta", "adamw"],
                        default="adamw")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--use_lr_scheduler", type=int, choices=[0, 1], default=1,
                        help="Use lr scheduler to reduce the learning rate during training")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout")
    parser.add_argument("--max_seq_length", type=int, default=4000)
    parser.add_argument("--min_word_frequency", type=int, default=-1)

    parser.add_argument("--embedding_mode", type=str, default="non_static",
                        choices=["static", "non_static"],
                        help="The mode to init embeddings:"
                             "\n\t1. rand: initialise the embedding randomly"
                             "\n\t2. static: using pretrained embeddings"
                             "\n\t2. non_static: using pretrained embeddings with fine tuning"
                             "\n\t2. multichannel: using both static and non-static modes")

    parser.add_argument('--embedding_size', type=int, default=100)
    parser.add_argument('--word_embedding_file', type=str, default='data/embeddings/word2vec_sg0_100.model')
    parser.add_argument("--attention_mode", type=str, choices=["text_label", "laat", "caml"], default="text_label")
    parser.add_argument("--d_a", type=int, help="The dimension of the first dense layer for self attention",
                        default=256)
    args = parser.parse_args()
    return args



