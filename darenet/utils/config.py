import argparse

parser = argparse.ArgumentParser(description='DARE-Net: Diagnosis-Aware Routing Mixture-of-Experts for Brain Age Estimation')
# =========== save path ================ #
parser.add_argument('--model_name'     ,default='best_model.pth.tar'    ,type=str, help="Checkpoint file name")
parser.add_argument('--output_dir'     ,default='./model/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
parser.add_argument('--train_folder'   ,default='../data/train'         ,type=str, help="Train set data path ")
parser.add_argument('--valid_folder'   ,default='../data/valid'         ,type=str, help="Validation set data path ")
parser.add_argument('--test_folder'    ,default='../data/test'          ,type=str, help="Test set data path ")
parser.add_argument('--excel_path'     ,default='../lables/Training.xls',type=str, help="Excel file path ")
parser.add_argument('--first_stage_net',default='./model/best.pth.tar'  ,type=str, help="When training the second stage network, appoint the trained first stage network checkpoint file path is needed ")
parser.add_argument('--npz_name'       ,default='test.npz'              ,type=str, help="After inference the trained model in test set, a npz file will be saved in assigned path. So the npz name need to be appointed. ")
parser.add_argument('--plot_name'      ,default='test.png'              ,type=str, help="After inference the trained model in test set, a scatter plot will be saved in assigned path. So the plot name need to be appointed. ")

#=========== hyperparameter ================ #
parser.add_argument('--model'       ,default='DARENet',type=str,   help="Deep learning model: DARENet (DARE-Net MTL) or ACDense (single-task)")
parser.add_argument('--num_workers' ,default=8           ,type=int,   help="The number of worker for dataloader")
parser.add_argument('--batch_size'  ,default=8           ,type=int,   help="Batch size during training process")
parser.add_argument('--epochs'      ,default=100         ,type=int,   help="Total training epochs")
parser.add_argument('--lr'          ,default=1e-3        ,type=float, help="Initial learning rate")
parser.add_argument('--schedular', type=str, default='cosine',help='choose the scheduler')
parser.add_argument('--print_freq'  ,default=40           ,type=int,   help="Training log print interval")
parser.add_argument('--weight_decay',default=5e-4        ,type=float, help="L2 weight decay ")
parser.add_argument('--use_gender'  ,default=True        ,type=bool,  help="If use sex label during training")
parser.add_argument('--dis_range'   ,default=5           ,type=int,   help="Discritize step when training the second stage network")

# warmup and cosine_lr_scheduler
parser.add_argument('--warmup_lr_init', type=float, default=5e-7 )
parser.add_argument('--warmup_epoch', type=int, default=0)
parser.add_argument('--min_lr', type=float, default=5e-6 )

# =========== loss function ================ #
parser.add_argument('--loss',       default='mse'       ,type=str,     help="Main loss fuction for training network")
parser.add_argument('--aux_loss',   default='ranking'   ,type=str,     help="Auxiliary loss function for training network")
parser.add_argument('--lbd',        default=10          ,type=float,   help="The weight between main loss function and auxiliary loss function")
parser.add_argument('--beta',       default=1           ,type=float,   help="The weight between ranking loss function and age difference loss function")
parser.add_argument('--sorter',     default='./darenet/Sodeep_pretrain_weight/Tied_rank_best_lstmla_slen_16.pth.tar', type=str,   help="When use ranking, the pretrained SoDeep sorter network weight need to be appointed")

# =========== MoE (Mixture of Experts) for MTL ================ #
parser.add_argument('--use_moe',         action='store_true', default=False,  help="Enable Mixture of Experts for age regression head")
parser.add_argument('--moe_num_experts', type=int,   default=6,     help="Number of expert networks in MoE (recommended: 4-8)")
parser.add_argument('--moe_topk',        type=int,   default=2,     help="Number of experts to activate per sample (Top-K routing)")
parser.add_argument('--moe_gate_temp',   type=float, default=1.2,   help="Temperature for gating softmax (0.8-1.5, lower=sharper)")
parser.add_argument('--moe_alpha',       type=float, default=0.02,  help="Weight for load balancing loss (0.01-0.05)")
parser.add_argument('--moe_use_dx',      action='store_true', default=False, help="Use diagnosis probability to guide gating")
parser.add_argument('--moe_entropy_w',   type=float, default=0.0,   help="Weight for gating entropy regularization (optional, 1e-3)")
parser.add_argument('--moe_tf_epochs',   type=int,   default=20,    help="Teacher forcing epochs for diagnosis-guided gating")

# =========== HPO (Hyperparameter Optimization) Support ================ #
parser.add_argument('--objective',       type=str, choices=['mae', 'combined'], default='mae', help="Optimization objective: 'mae' (MAE only) or 'combined' (MAE + Acc)")
parser.add_argument('--acc_weight',      type=float, default=0.2,   help="Weight for accuracy in combined objective (not used if objective='mae')")
parser.add_argument('--trial_id',        type=str,   default='',    help="Trial identifier for HPO (injected by hpo script)")
parser.add_argument('--save_all_ckpts',  action='store_true', default=False, help="Save checkpoint at every epoch (for HPO analysis)")

# =========== Advanced Learning Rate Scheduler ================ #
parser.add_argument('--use_timm_sched',  action='store_true', default=False, help="Use timm-style learning rate scheduler")
parser.add_argument('--lr_decay_epochs', type=int,   default=30,    help="Epochs for step decay (if schedular='step')")
parser.add_argument('--lr_decay_rate',   type=float, default=0.1,   help="Decay rate for step scheduler")

# =========== Pairwise Order Loss & Heteroscedastic Regression ================ #
parser.add_argument('--pairwise_w',  type=float, default=0.0, help='Weight for pairwise order loss (0 disables)')
parser.add_argument('--pair_delta',  type=float, default=2.0, help='Min age diff for pairwise selection')
parser.add_argument('--age_hetero',  action='store_true', default=False, help='Enable heteroscedastic age regression')

args = parser.parse_args()
opt = args 