import os, torch, json, csv
import datetime, warnings
import tensorboardX
import numpy as np
import torch.nn as nn
from utils.config import opt
from utils.lr_scheduler import build_scheduler
from model import ACDense
from model.ranking_loss import rank_difference_loss
from load_data import IMG_Folder,Integer_Multiple_Batch_Size
from prediction_first_stage_mtl import test_mtl
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings("ignore")

# Check if using MTL (Multi-Task Learning) mode - DARE-Net
# Note: This is set at module load time, but HPO may change opt.model later
# So we re-check in main() function
USE_MTL = (opt.model == 'DARENet' or opt.model == 'ScaleDenseMTL')


def get_dynamic_ranking_weight(epoch, base_lbd, warmup_start=10, warmup_end=20, max_lbd=2.0):
    """
    Compute dynamic ranking loss weight with warmup schedule
    
    Args:
        epoch (int): current training epoch
        base_lbd (float): base lambda value from config
        warmup_start (int): epoch to start ranking loss warmup (default: 10)
        warmup_end (int): epoch to end ranking loss warmup (default: 20)
        max_lbd (float): maximum lambda value (default: 2.0)
    
    Returns:
        float: current ranking loss weight
    """
    if epoch < warmup_start:
        return 0.0
    elif epoch < warmup_end:
        return base_lbd * (epoch - warmup_start) / (warmup_end - warmup_start)
    else:
        return min(base_lbd, max_lbd)


def get_classification_warmup_factor(epoch, warmup_epochs=10):
    """
    Compute classification loss warmup factor to prevent early gradient dominance
    
    Args:
        epoch (int): current training epoch
        warmup_epochs (int): number of epochs for warmup (default: 10)
    
    Returns:
        float: warmup factor in [0, 1], linearly increasing from 0 to 1
    """
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    else:
        return 1.0


def pairwise_order_loss(age_pred, age_true, margin=0.0, delta=2.0):
    """
    Pairwise order loss: encourages correct age ordering between samples
    
    Args:
        age_pred (torch.Tensor): predicted ages [B, 1]
        age_true (torch.Tensor): true ages [B, 1]
        margin (float): margin for hinge loss (default: 0.0)
        delta (float): minimum age difference for pair selection (default: 2.0)
    
    Returns:
        torch.Tensor: pairwise order loss (scalar)
    """
    ap = age_pred.view(-1)
    at = age_true.view(-1)
    n = ap.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=ap.device)
    
    # Generate all pairwise combinations
    idx = torch.combinations(torch.arange(n, device=ap.device), r=2)  # [M, 2]
    da = at[idx[:, 0]] - at[idx[:, 1]]
    dp = ap[idx[:, 0]] - ap[idx[:, 1]]
    
    # Only consider pairs with significant age difference
    mask = da.abs() > delta
    if mask.sum() == 0:
        return torch.tensor(0.0, device=ap.device)
    
    # Hinge loss: encourage correct ordering
    target = da.sign()
    return torch.nn.functional.relu(-dp * target + margin)[mask].mean()


def prepare_gender_input(gender, device):
    """
    Convert gender labels to one-hot encoding using torch.nn.functional.one_hot
    
    Args:
        gender (torch.Tensor): gender labels (0 or 1)
        device (torch.device): target device
    
    Returns:
        torch.Tensor: one-hot encoded gender [batch_size, 2]
    """
    return torch.nn.functional.one_hot(gender.long(), num_classes=2).float().to(device)


def prepare_target(target, device):
    """
    Prepare target tensor for regression task
    
    Args:
        target (np.ndarray or torch.Tensor): target values
        device (torch.device): target device
    
    Returns:
        torch.Tensor: prepared target tensor [batch_size, 1]
    """
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    return target.float().unsqueeze(1).to(device) if target.dim() == 1 else target.float().to(device)


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.enabled = True

 # ======== main function =========== #
print('===== hyper-parameter ====== ')
print("=> network     : {}".format(opt.model))
print("=> lambda      : {}".format(opt.lbd))
print("=> batch size  : {}".format(opt.batch_size))
print("=> learning rate    : {}".format(opt.lr))
print("=> weight decay     : {}".format(opt.weight_decay))
print("=> aux loss         : {}x{}".format(opt.aux_loss, opt.lbd))

def main(res=None):
    # CRITICAL: Re-check USE_MTL based on current opt.model
    # HPO may have changed opt.model after module import
    global USE_MTL
    USE_MTL = (opt.model == 'DARENet' or opt.model == 'ScaleDenseMTL')
    
    best_metric = 100
    best_mae = 100
    best_acc = None
    best_epoch = 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Fix: Prevent "None" file creation when HPO mode (res=None)
    if res is None:
        res = os.path.join(opt.output_dir, 'result.txt')

    # Ensure output directory exists
    os.makedirs(opt.output_dir, exist_ok=True)
    
    json_path = os.path.join(opt.output_dir,'hyperparameter.json')
    with open(json_path,'w') as f:
        f.write(json.dumps(vars(opt)
                            ,ensure_ascii=False
                            ,indent=4
                            ,separators=(',', ':')))
    print("=========== start train the brain age estimation model =========== \n")
    print(" ==========> Using {} processes for data loader.".format(opt.num_workers))
    
    # =========== HPO: Create epoch-level metrics CSV =========== #
    # Note: csv is already imported at the top of the file
    epoch_csv = os.path.join(opt.output_dir, 'metrics_epoch.csv')
    if not os.path.exists(epoch_csv):
        with open(epoch_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            if USE_MTL:
                writer.writerow(['epoch', 'train_loss', 'train_mae', 'train_acc', 'train_loss_age', 'train_loss_cls', 'train_loss_rank',
                               'valid_loss', 'valid_mae', 'valid_acc', 'valid_loss_age', 'valid_loss_cls', 'valid_loss_rank',
                               'lr', 'uncertainty_age', 'uncertainty_cls', 'lbd_rank', 'best_metric', 'is_best'])
            else:
                writer.writerow(['epoch', 'train_loss', 'train_mae', 'train_loss1', 'train_loss2',
                               'valid_loss', 'valid_mae', 'valid_loss1', 'valid_loss2', 
                               'lr', 'best_metric', 'is_best'])
    
    # Create detailed training log file
    train_log = os.path.join(opt.output_dir, 'training_log.txt')
    with open(train_log, 'w') as f:
        f.write('=' * 80 + '\n')
        f.write(f'Training Started at {datetime.datetime.now()}\n')
        f.write('=' * 80 + '\n\n')
        f.write('=== Hyperparameters ===\n')
        for key, value in sorted(vars(opt).items()):
            f.write(f'  {key}: {value}\n')
        f.write('\n' + '=' * 80 + '\n\n')


    # ===========  define data folder and CUDA device =========== #
    train_data = Integer_Multiple_Batch_Size(IMG_Folder(opt.excel_path, opt.train_folder, use_diagnosis=USE_MTL), opt.batch_size)
    # Validation data: DO NOT pad to batch size (for fair evaluation in HPO)
    valid_data = IMG_Folder(opt.excel_path, opt.valid_folder, use_diagnosis=USE_MTL)
    
    # ===========  define data loader with stratified sampling for MTL =========== #
    # Note: WeightedRandomSampler can boost Dx_Acc but may hurt MAE by distorting age distribution
    # Recommendation: Start with shuffle=True (sampler disabled), enable sampler only after MAE stabilizes
    USE_WEIGHTED_SAMPLER = False  # Set to True to enable balanced sampling (may increase Dx_Acc but risk higher MAE)
    
    # Always compute class distribution for MTL (for dynamic class weights in loss function)
    class_sample_counts = None
    class_weights_loss = None  # For loss function
    train_dx = None  # Store diagnosis labels for potential WeightedRandomSampler use
    
    if USE_MTL:
        # Extract diagnosis labels from CSV/Excel to compute actual class distribution
        try:
            print('=> Computing class distribution from training data...')
            img_folder = train_data.folder_dataset
            table_refer = img_folder.table_refer  # DataFrame with metadata
            sub_fns = img_folder.sub_fns  # List of filenames
            dx_map = img_folder.dx_map  # Diagnosis mapping
            
            train_dx = []
            for sub_fn in sub_fns:
                # Normalize filename for matching
                # Handle: ADNI_AD_sub-0001_nonlin_brain.nii.gz -> ADNI_AD_sub-0001
                sub_fn_base = sub_fn.replace('.nii.gz', '').replace('.nii', '')
                if sub_fn_base.endswith('_nonlin_brain'):
                    sub_fn_base = sub_fn_base.replace('_nonlin_brain', '')
                
                # Match filename to table entry (same logic as IMG_Folder.__getitem__)
                dx_label = 1  # default to MCI
                for idx, row in table_refer.iterrows():
                    # Get ID from first column or 'ID' column
                    if 'ID' in table_refer.columns:
                        sid = str(row['ID'])
                    else:
                        sid = str(row.iloc[0])
                    
                    # Normalize CSV ID for matching
                    sid_base = sid.replace('.nii.gz', '').replace('.nii', '')
                    
                    if sid_base == sub_fn_base:
                        # Extract diagnosis - support both OASIS (dx column) and ADNI (source_dataset column)
                        if 'dx' in row:
                            # OASIS format: dx column contains 0, 1, or 2 directly
                            dx_label = int(row['dx'])
                            if dx_label not in [0, 1, 2]:
                                print(f'Warning: Invalid dx value {dx_label} for {sid_base}, defaulting to MCI')
                                dx_label = 1
                        elif 'source_dataset' in row:
                            # ADNI format: source_dataset column contains string like 'ADNI_Norm', 'ADNI_MCI', 'ADNI_AD'
                            dx_str = row['source_dataset']
                            dx_label = dx_map.get(dx_str, 1)
                        elif 'diagnosis' in row:
                            dx_label = int(row['diagnosis'])
                        else:
                            # No diagnosis column found, default to MCI
                            dx_label = 1
                        break
                train_dx.append(dx_label)
            
            # Compute class distribution: [Norm, MCI, AD]
            class_sample_counts = [train_dx.count(0), train_dx.count(1), train_dx.count(2)]
            total_samples = sum(class_sample_counts)
            
            if all(c > 0 for c in class_sample_counts):
                # All classes present: use standard inverse frequency weighting
                # Formula: weight_i = N_total / (K * n_i) where K=number of classes
                class_weights_loss = [total_samples / (3 * c) for c in class_sample_counts]
                
                print(f'=> Class distribution computed: Norm={class_sample_counts[0]}, MCI={class_sample_counts[1]}, AD={class_sample_counts[2]}')
                print(f'=> Dynamic class weights for loss (Norm/MCI/AD): {[f"{w:.3f}" for w in class_weights_loss]}')
            else:
                # Some classes missing: compute weights for present classes, use small weight for missing ones
                print('=> Warning: Some diagnosis classes missing in training data')
                print(f'=> Class counts: Norm={class_sample_counts[0]}, MCI={class_sample_counts[1]}, AD={class_sample_counts[2]}')
                
                # Compute weights for present classes
                present_classes = [i for i, c in enumerate(class_sample_counts) if c > 0]
                if len(present_classes) > 0:
                    # Use inverse frequency for present classes
                    present_counts = [class_sample_counts[i] for i in present_classes]
                    present_weights = [total_samples / (len(present_classes) * c) for c in present_counts]
                    
                    # Assign weights: present classes get computed weights, missing classes get small weight
                    class_weights_loss = []
                    for i in range(3):
                        if i in present_classes:
                            idx = present_classes.index(i)
                            class_weights_loss.append(present_weights[idx])
                        else:
                            # Missing class gets small weight (1/10 of average present weight)
                            avg_weight = sum(present_weights) / len(present_weights)
                            class_weights_loss.append(avg_weight / 10.0)
                    
                    print(f'=> Computed weights for present classes: {[f"{class_weights_loss[i]:.3f}" for i in present_classes]}')
                    print(f'=> Using small weights for missing classes: {[f"{class_weights_loss[i]:.3f}" for i in range(3) if i not in present_classes]}')
                    print(f'=> Final class weights (Norm/MCI/AD): {[f"{w:.3f}" for w in class_weights_loss]}')
                    print('=> Note: Model may struggle with missing classes. Consider checking dataset split.')
                else:
                    # No valid classes found, fall back to default
                    print('=> Error: No valid diagnosis classes found!')
                    print('=> Falling back to default ADNI class weights')
                    class_sample_counts = None
                    class_weights_loss = None
                    train_dx = None
        except Exception as e:
            print(f'=> Warning: Failed to compute class distribution: {e}')
            print('=> Falling back to default ADNI class weights')
            import traceback
            traceback.print_exc()
            class_sample_counts = None
            class_weights_loss = None
            train_dx = None
    
    # Create weighted sampler if enabled
    if USE_MTL and USE_WEIGHTED_SAMPLER and train_dx is not None and class_weights_loss is not None:
        try:
            # Use the already computed class_weights_loss for sampling
            class_weights_sampling = class_weights_loss
            
            # Extend weights to match Integer_Multiple_Batch_Size padding
            sample_weights = []
            for idx in train_data.complemented_idx:
                sample_weights.append(class_weights_sampling[train_dx[idx]])
            
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            print(f'=> Using WeightedRandomSampler for balanced training')
            print(f'=> Sampling weights (Norm/MCI/AD): {[f"{w:.3f}" for w in class_weights_sampling]}')
            use_sampler = True
        except Exception as e:
            print(f'=> Warning: Failed to create WeightedRandomSampler: {e}')
            print('=> Falling back to shuffle=True')
            import traceback
            traceback.print_exc()
            sampler = None
            use_sampler = False
    else:
        sampler = None
        use_sampler = False

    # ===========  define data loader =========== #
    train_loader = torch.utils.data.DataLoader(  train_data
                                                ,batch_size=opt.batch_size
                                                ,num_workers=opt.num_workers
                                                ,pin_memory=True
                                                ,drop_last=False
                                                ,shuffle=(not use_sampler)
                                                ,sampler=sampler if use_sampler else None
                                                )
    valid_loader = torch.utils.data.DataLoader(  valid_data
                                                ,batch_size=opt.batch_size 
                                                ,num_workers=opt.num_workers 
                                                ,pin_memory=True
                                                ,drop_last=False
                                                )
    
    # ===========  build and set model  =========== #  
    if opt.model == 'ACDense':
        model = ACDense.ACDense(8, 5, opt.use_gender)
    elif opt.model == 'DARENet' or opt.model == 'ScaleDenseMTL':
        model = ACDense.DARENet(8, 5, opt.use_gender, num_classes=3, opt=opt)
        print('=> Using DARE-Net: Diagnosis-Aware Routing MoE for brain age + dementia staging')
        if getattr(opt, 'use_moe', False):
            print(f'=> MoE enabled: {opt.moe_num_experts} experts, Top-{opt.moe_topk} routing')
            if getattr(opt, 'moe_use_dx', False):
                print(f'=> Diagnosis-guided gating with {opt.moe_tf_epochs} epochs teacher forcing')
    else:
        raise ValueError('Wrong model choose')

    model.apply(weights_init)
    model = nn.DataParallel(model).to(device)
    model_test = model

    # =========== define the loss function =========== #
    loss_func_dict = {'mae': nn.L1Loss().to(device)
                     ,'mse': nn.MSELoss().to(device)
                     ,'smoothl1': nn.SmoothL1Loss().to(device)
                     ,'ranking':rank_difference_loss(sorter_checkpoint_path=opt.sorter
                                                    ,beta=opt.beta).to(device)
                     }
        
    criterion1 = loss_func_dict[opt.loss]  # Age regression loss
    criterion2 = loss_func_dict[opt.aux_loss]  # Ranking loss
    
    # Multi-Task Learning: Add classification loss with class weights and label smoothing
    if USE_MTL:
        # Use dynamically computed class weights if available, otherwise fall back to ADNI defaults
        if class_weights_loss is not None:
            class_weights = torch.tensor(class_weights_loss, dtype=torch.float).to(device)  # [Norm, MCI, AD]
            print('=> Using dynamically computed class weights from training data')
        else:
            # Fallback to ADNI class weights: Norm(535):MCI(1005):AD(322), Total=1862
            # weights = N_total / (K * n_i) = 1862 / (3 * n_i)
            class_weights = torch.tensor([1.160, 0.618, 1.929], dtype=torch.float).to(device)  # [Norm, MCI, AD]
            print('=> Using default ADNI class weights (Norm/MCI/AD): [1.160, 0.618, 1.929]')
            print('=> Note: These weights are based on ADNI distribution. Consider checking your dataset distribution.')
        
        # Use label_smoothing=0.05 to prevent overconfidence and improve generalization
        criterion_ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05).to(device)
        print('=> Using class-weighted CrossEntropy with label_smoothing=0.05 for diagnosis classification')
        print('=> Class weights (Norm/MCI/AD): {}'.format(class_weights.cpu().numpy()))

    # =========== define optimizer and learning rate scheduler =========== #
    # For MTL, also optimize the uncertainty weights (log_vars)
    if USE_MTL:
        optimizer = torch.optim.Adam([
            {'params': [p for n, p in model.named_parameters() if 'log_vars' not in n]},
            {'params': [p for n, p in model.named_parameters() if 'log_vars' in n], 'lr': opt.lr}
        ], lr=opt.lr, weight_decay=opt.weight_decay, amsgrad=True)
        print('=> Optimizer includes uncertainty weights for task balancing')
    else:
        optimizer = torch.optim.Adam( model.parameters()
                                     ,lr=opt.lr
                                     ,weight_decay=opt.weight_decay
                                     ,amsgrad=True
                                    )
    
    # Choose scheduler based on --use_timm_sched flag
    if getattr(opt, 'use_timm_sched', False):
        lr_scheduler = build_scheduler(opt, optimizer, n_iter_per_epoch=len(train_loader))
        scheduler = None
        print('=> Using timm-style LR scheduler:', opt.schedular)
    else:
        lr_scheduler = None
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, verbose=1, patience=5, factor=0.5
        )
    
    early_stopping = EarlyStopping(patience=20, verbose=True)
    
    # =========== define tensorboardX and show traing start signal =========== #
    saved_metrics, saved_epos = [], []
    num_epochs = int(opt.epochs)
    sum_writer = tensorboardX.SummaryWriter(opt.output_dir)
    print(" ==========> Training is getting started...")
    print(" ==========> Training takes {} epochs.".format(num_epochs))

    # =========== start train =========== #
    for epoch in range(opt.epochs):
            
        # ===========  train for one epoch   =========== #
        if USE_MTL:
            train_results = train(  train_loader=train_loader
                                          , model=model
                                          , criterion1=criterion1
                                          , criterion2=criterion2
                                          , criterion_ce=criterion_ce
                                          , optimizer=optimizer
                                          , device=device
                                          , epoch=epoch)
            train_loss, train_mae, train_acc = train_results[:3]
            train_loss_age, train_loss_cls, train_loss_rank = train_results[3:6]
        else:
            train_results = train(  train_loader=train_loader
                                          , model=model
                                          , criterion1=criterion1
                                          , criterion2=criterion2
                                          , optimizer=optimizer
                                          , device=device
                                          , epoch=epoch)
            train_loss, train_mae = train_results[:2]
            train_loss1, train_loss2 = train_results[2:4]

        # ===========  evaluate on validation set ===========  #
        if USE_MTL:
            valid_results = validate( valid_loader=valid_loader
                                            , model=model 
                                            , criterion1=criterion1
                                            , criterion2=criterion2
                                            , criterion_ce=criterion_ce
                                            , device=device
                                            , epoch=epoch)
            valid_loss, valid_mae, valid_acc = valid_results[:3]
            valid_loss_age, valid_loss_cls, valid_loss_rank = valid_results[3:6]
        else:
            valid_results = validate( valid_loader=valid_loader
                                            , model=model 
                                            , criterion1=criterion1
                                            , criterion2=criterion2
                                            , device=device
                                            , epoch=epoch)
            valid_loss, valid_mae = valid_results[:2]
            valid_loss1, valid_loss2 = valid_results[2:4]        

        # ===========  learning rate decay =========== #  
        if lr_scheduler is not None:
            lr_scheduler.step(epoch)
        else:
            scheduler.step(valid_mae)
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"\n*learning rate {cur_lr:.2e}*\n")

        # ===========  write in tensorboard scaler =========== #
        sum_writer.add_scalar('train/loss', train_loss, epoch)
        sum_writer.add_scalar('train/mae', train_mae, epoch)
        sum_writer.add_scalar('valid/loss', valid_loss, epoch)
        sum_writer.add_scalar('valid/mae', valid_mae, epoch)
        sum_writer.add_scalar('lr', cur_lr, epoch)
        
        if USE_MTL:
            sum_writer.add_scalar('train/dx_acc', train_acc, epoch)
            sum_writer.add_scalar('valid/dx_acc', valid_acc, epoch)
            sum_writer.add_scalar('train/loss_age', train_loss_age, epoch)
            sum_writer.add_scalar('train/loss_cls', train_loss_cls, epoch)
            sum_writer.add_scalar('train/loss_rank', train_loss_rank, epoch)
            sum_writer.add_scalar('valid/loss_age', valid_loss_age, epoch)
            sum_writer.add_scalar('valid/loss_cls', valid_loss_cls, epoch)
            sum_writer.add_scalar('valid/loss_rank', valid_loss_rank, epoch)
            
            # Log uncertainty weights (exp(-log_vars) are the actual task weights)
            log_vars = model.module.log_vars if isinstance(model, nn.DataParallel) else model.log_vars
            uncertainty_age = torch.exp(-log_vars[0]).item()
            uncertainty_cls = torch.exp(-log_vars[1]).item()
            sum_writer.add_scalar('train/uncertainty_age', uncertainty_age, epoch)
            sum_writer.add_scalar('train/uncertainty_cls', uncertainty_cls, epoch)
            # Log the raw log_vars as well for debugging
            sum_writer.add_scalar('train/log_var_age', log_vars[0].item(), epoch)
            sum_writer.add_scalar('train/log_var_cls', log_vars[1].item(), epoch)
            
            # Get current ranking loss weight
            lbd_current = get_dynamic_ranking_weight(epoch, opt.lbd, warmup_start=10, warmup_end=20, max_lbd=2.0)
            sum_writer.add_scalar('train/lbd_rank', lbd_current, epoch)
        else:
            sum_writer.add_scalar('train/loss1', train_loss1, epoch)
            sum_writer.add_scalar('train/loss2', train_loss2, epoch)
            sum_writer.add_scalar('valid/loss1', valid_loss1, epoch)
            sum_writer.add_scalar('valid/loss2', valid_loss2, epoch)

        # ===========  record the  best metric and save checkpoint ===========  #
        # HPO: Use opt.objective to determine best model criterion
        if USE_MTL and opt.objective == 'combined':
            # Combined metric: MAE + weight * (1 - Acc)
            valid_metric = valid_mae + float(opt.acc_weight) * (1.0 - valid_acc)
            print(f'Combined valid metric: MAE={valid_mae:.4f}, DX_Acc={valid_acc:.4f}, '
                  f'Combined={valid_metric:.4f} (lower is better)')
        else:
            # Default: only MAE (for HPO, this is the target)
            valid_metric = valid_mae
            
        is_best = False
        if valid_metric < best_metric:
            is_best = True
            best_metric = min(valid_metric, best_metric)
            best_mae = float(valid_mae)
            best_acc = float(valid_acc) if USE_MTL else None
            best_epoch = epoch
                
            saved_metrics.append(valid_metric)
            saved_epos.append(epoch)
            if USE_MTL:
                print('=======>   Best at epoch %d, valid metric %f (MAE: %f, DX_Acc: %f)\n' % (epoch, best_metric, valid_mae, valid_acc))
            else:
                print('=======>   Best at epoch %d, valid MAE %f\n' % (epoch, best_metric))
        
        # =========== HPO: Write epoch metrics to CSV =========== #
        with open(epoch_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if USE_MTL:
                writer.writerow([epoch, float(train_loss), float(train_mae), float(train_acc), 
                               float(train_loss_age), float(train_loss_cls), float(train_loss_rank),
                               float(valid_loss), float(valid_mae), float(valid_acc), 
                               float(valid_loss_age), float(valid_loss_cls), float(valid_loss_rank),
                               cur_lr, uncertainty_age, uncertainty_cls, lbd_current, float(best_metric), int(is_best)])
            else:
                writer.writerow([epoch, float(train_loss), float(train_mae), float(train_loss1), float(train_loss2),
                               float(valid_loss), float(valid_mae), float(valid_loss1), float(valid_loss2),
                               cur_lr, float(best_metric), int(is_best)])
        
        # =========== Write detailed training log =========== #
        with open(train_log, 'a') as f:
            f.write(f'Epoch {epoch}/{opt.epochs-1}\n')
            f.write(f'  Time: {datetime.datetime.now()}\n')
            f.write(f'  Learning Rate: {cur_lr:.6e}\n')
            if USE_MTL:
                f.write(f'  Train - Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, DX_Acc: {train_acc:.4f}\n')
                f.write(f'          Age Loss: {train_loss_age:.4f}, Cls Loss: {train_loss_cls:.4f}, Rank Loss: {train_loss_rank:.4f}\n')
                f.write(f'  Valid - Loss: {valid_loss:.4f}, MAE: {valid_mae:.4f}, DX_Acc: {valid_acc:.4f}\n')
                f.write(f'          Age Loss: {valid_loss_age:.4f}, Cls Loss: {valid_loss_cls:.4f}, Rank Loss: {valid_loss_rank:.4f}\n')
                f.write(f'  Task Weights - Age: {uncertainty_age:.4f}, Cls: {uncertainty_cls:.4f}\n')
                f.write(f'  Ranking Loss Weight (lambda): {lbd_current:.4f}\n')
            else:
                f.write(f'  Train - Loss: {train_loss:.4f}, MAE: {train_mae:.4f}\n')
                f.write(f'          Loss1: {train_loss1:.4f}, Loss2: {train_loss2:.4f}\n')
                f.write(f'  Valid - Loss: {valid_loss:.4f}, MAE: {valid_mae:.4f}\n')
                f.write(f'          Loss1: {valid_loss1:.4f}, Loss2: {valid_loss2:.4f}\n')
            f.write(f'  Best Metric: {best_metric:.4f} (epoch {best_epoch})\n')
            if is_best:
                f.write(f'  *** NEW BEST MODEL ***\n')
            f.write('\n')

        save_checkpoint({'epoch': epoch
                        ,'arch': opt.model
                        ,'state_dict': model.state_dict()
                        ,'optimizer': optimizer.state_dict()}
                        , is_best
                        , opt.output_dir
                        , model_name=opt.model
                        )

        # ===========  early_stopping needs the validation loss or MAE to check if it has decresed 
        early_stopping(valid_mae)        
        if early_stopping.early_stop:
            print("======= Early stopping =======")
            break

    # =========== write traning and validation log =========== #
    os.system('echo " ================================== "')
    os.system('echo " ==== TRAIN MAE mtc:{:.5f}" >> {}'.format(train_mae, res))
    
    # Write summary to detailed log
    with open(train_log, 'a') as f:
        f.write('=' * 80 + '\n')
        f.write(f'Training Completed at {datetime.datetime.now()}\n')
        f.write('=' * 80 + '\n\n')
        f.write(f'Best Model: Epoch {best_epoch}, Metric: {best_metric:.4f}\n')
        if USE_MTL:
            f.write(f'  MAE: {best_mae:.4f}, DX_Acc: {best_acc:.4f}\n')
        else:
            f.write(f'  MAE: {best_mae:.4f}\n')
        f.write('\nTop 10 Epochs (by metric):\n')
    
    print('Epo - Mtc')
    mtc_epo = dict(zip(saved_metrics, saved_epos))
    rank_mtc = sorted(mtc_epo.keys(), reverse=False)
    try:
        for i in range(10):
            epo = mtc_epo[rank_mtc[i]]
            mtc = rank_mtc[i]
            print('{:03} {:.3f}'.format(epo, mtc))
            os.system('echo "epo:{:03} mtc:{:.3f}" >> {}'.format(epo, mtc, res))
            with open(train_log, 'a') as f:
                f.write(f'  Epoch {epo:03d}: {mtc:.4f}\n')
    except:
        pass
    
    # ===========  clean up ===========  #
    torch.cuda.empty_cache()
    # =========== test the trained model on test dataset =========== #
    # Fix: Don't pad test data for fair evaluation (same as validation)
    test_data = IMG_Folder(opt.excel_path, opt.test_folder, use_diagnosis=USE_MTL)
    test_loader = torch.utils.data.DataLoader( test_data
                                              ,batch_size=opt.batch_size 
                                              ,num_workers=opt.num_workers 
                                              ,pin_memory=True
                                              ,drop_last=False)
    
    # =========== test on the best model on test data =========== # 
    model_best = model_test
    # PyTorch 2.6+ compatibility: set weights_only=False for checkpoint loading
    checkpoint_path = os.path.join(opt.output_dir+opt.model+'_best_model.pth.tar')
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions
        checkpoint = torch.load(checkpoint_path)
    model_best.load_state_dict(checkpoint['state_dict']) 
    print('========= best model test result ===========')
    
    # Use MTL test function (supports both single-task and multi-task models)
    test_results = test_mtl(test_loader, model_best, device, save_results=True)
    test_MAE = test_results['age_mae']
    
    if USE_MTL:
        test_dx_acc = test_results['dx_acc']
        print(f"Test MAE: {test_MAE:.4f}, Test DX_Acc: {test_dx_acc:.4f}")
        if res:
            os.system('echo " ================================== "')
            os.system('echo "best valid model TEST MAE mtc:{:.5f}" >> {}'.format(test_MAE, res))
            os.system('echo "best valid model TEST DX_Acc mtc:{:.5f}" >> {}'.format(test_dx_acc, res))
    else:
        print(f"Test MAE: {test_MAE:.4f}")
        if res:
            os.system('echo " ================================== "')
            os.system('echo "best valid model TEST MAE mtc:{:.5f}" >> {}'.format(test_MAE, res))
    
    # =========== HPO: Return results as dictionary =========== #
    result_dict = {
        'best_metric': float(best_metric),
        'best_mae': float(best_mae),
        'best_acc': float(best_acc) if USE_MTL else None,
        'best_epoch': int(best_epoch),
        'test_mae': float(test_MAE),
        'test_acc': float(test_dx_acc) if USE_MTL else None,
        'output_dir': opt.output_dir,
    }
    
    # Save metrics.json for HPO
    # Note: json is already imported at the top of the file
    with open(os.path.join(opt.output_dir, 'metrics.json'), 'w') as f:
        json.dump(result_dict, f, indent=2)
    
    # Write test results to detailed log
    with open(train_log, 'a') as f:
        f.write('\n' + '=' * 80 + '\n')
        f.write(f'Test Results (Best Model from Epoch {best_epoch})\n')
        f.write('=' * 80 + '\n')
        if USE_MTL:
            f.write(f'  Test MAE: {test_MAE:.4f}\n')
            f.write(f'  Test DX_Acc: {test_dx_acc:.4f}\n')
        else:
            f.write(f'  Test MAE: {test_MAE:.4f}\n')
        f.write('\nFinal Summary:\n')
        f.write(f'  Best Validation Metric: {best_metric:.4f} (Epoch {best_epoch})\n')
        f.write(f'  Best Validation MAE: {best_mae:.4f}\n')
        if USE_MTL:
            f.write(f'  Best Validation DX_Acc: {best_acc:.4f}\n')
        f.write(f'  Test MAE: {test_MAE:.4f}\n')
        if USE_MTL:
            f.write(f'  Test DX_Acc: {test_dx_acc:.4f}\n')
        f.write('\n' + '=' * 80 + '\n')
        f.write(f'All results saved to: {opt.output_dir}\n')
        f.write('=' * 80 + '\n')
    
    return result_dict

def train(train_loader, model, criterion1, criterion2, optimizer, device, epoch, criterion_ce=None):
    '''
    For training process (supports both single-task and multi-task learning)

    Args:
        train_loader (data loader): train data loader.
        model (CNN model): convolutional neural network.
        criterion1 (loss fucntion): main loss function (age regression).
        criterion2 (loss fucntion): aux loss function (ranking).
        optimizer (torch.optimizer): optimizer which is defined in 'main'
        device (torch device type): default: GPU
        epoch (int): training epoch idex
        criterion_ce (loss function): classification loss for MTL mode

    Returns:
        [float]: training loss average, MAE average, and optionally classification accuracy
    '''
    
    losses = AverageMeter()
    MAE = AverageMeter()
    LOSS1 = AverageMeter()
    LOSS2 = AverageMeter()
    LOSS_CLS = AverageMeter()
    DX_ACC = AverageMeter()
    
    is_mtl = (criterion_ce is not None)
    
    # Use unified dynamic ranking loss weight function
    if is_mtl:
        lbd_current = get_dynamic_ranking_weight(epoch, opt.lbd, warmup_start=10, warmup_end=20, max_lbd=2.0)
        cls_warmup = get_classification_warmup_factor(epoch, warmup_epochs=10)
        if epoch == 0 or epoch == 5 or epoch == 10 or epoch == 20:
            print(f'=> Epoch {epoch}: ranking loss weight (lbd) = {lbd_current:.3f}, cls warmup = {cls_warmup:.3f}')
    else:
        lbd_current = opt.lbd
        cls_warmup = 1.0

    for i, batch in enumerate(train_loader):
        if is_mtl:
            img, _, target, male, dx_label = batch
            dx_label = dx_label.to(device).long()
        else:
            img, _, target, male = batch

        # =========== prepare input data using standardized functions =========== #
        input = img.to(device)
        target = prepare_target(target, device)  # Unified target processing
        
        # Convert gender to one-hot encoding using torch.nn.functional.one_hot
        # Always prepare male input (even if use_gender=False) to avoid DataParallel issues
        male = prepare_gender_input(male, device)  # Unified gender one-hot encoding

        # =========== compute output and loss =========== #
        model.train()
        model.zero_grad()
        
        if is_mtl:
            # MTL forward: pass dx_label and epoch for MoE teacher forcing
            age_pred, cls_logits = model(input, male, dx_label if getattr(opt, 'use_moe', False) and getattr(opt, 'moe_use_dx', False) else None, epoch)
            
            # Age regression loss (heteroscedastic if enabled)
            if getattr(opt, 'age_hetero', False):
                lv = getattr(model.module, 'age_logvar', None) if isinstance(model, nn.DataParallel) else getattr(model, 'age_logvar', None)
                if lv is not None:
                    loss_age = 0.5 * torch.exp(-lv) * (age_pred - target)**2 + 0.5 * lv
                    loss_age = loss_age.mean()
                else:
                    loss_age = criterion1(age_pred, target)
            else:
                loss_age = criterion1(age_pred, target)
            
            # Classification loss with warmup (gradual introduction to prevent early dominance)
            loss_cls_raw = criterion_ce(cls_logits, dx_label)
            loss_cls = cls_warmup * loss_cls_raw  # Apply warmup factor
            
            # Ranking loss (integrated into age task)
            if lbd_current > 0:
                loss_rank = criterion2(age_pred, target)
                # Combine age regression loss and ranking loss as a single age task
                loss_age_combined = loss_age + lbd_current * loss_rank
            else:
                loss_rank = torch.tensor(0.0).to(device)
                loss_age_combined = loss_age
            
            # Pairwise order loss（可选）
            if getattr(opt, 'pairwise_w', 0.0) > 0:
                loss_pair = pairwise_order_loss(age_pred, target, margin=0.0, delta=getattr(opt, 'pair_delta', 2.0))
                loss_age_combined = loss_age_combined + getattr(opt, 'pairwise_w', 0.0) * loss_pair
            else:
                loss_pair = torch.tensor(0.0, device=device)
            
            # Uncertainty-weighted MTL loss with 2 tasks (age+ranking, classification)
            # Clamp log_vars to [-5, 5] to prevent negative total loss
            log_vars = model.module.log_vars if isinstance(model, nn.DataParallel) else model.log_vars
            log_vars_clamped = torch.clamp(log_vars, -5.0, 5.0)
            
            # Multi-task loss: L = exp(-s_i) * L_i + s_i
            # This ensures the total loss is always positive
            loss = (torch.exp(-log_vars_clamped[0]) * loss_age_combined + log_vars_clamped[0] +
                    torch.exp(-log_vars_clamped[1]) * loss_cls + log_vars_clamped[1])
            
            # Add MoE auxiliary losses if MoE is enabled
            if getattr(opt, 'use_moe', False):
                moe_aux = getattr(model.module, 'moe_aux', None) if isinstance(model, nn.DataParallel) else getattr(model, 'moe_aux', None)
                if moe_aux is not None:
                    # Load balancing loss
                    loss = loss + opt.moe_alpha * moe_aux['balance']
                    # Optional: entropy regularization
                    if getattr(opt, 'moe_entropy_w', 0.0) > 0:
                        loss = loss - opt.moe_entropy_w * moe_aux['entropy']
            
            # Compute metrics
            mae = metric(age_pred.detach(), target.detach().cpu())
            dx_acc = (cls_logits.argmax(1) == dx_label).float().mean().item()
            
            losses.update(loss.item(), img.size(0))
            LOSS1.update(loss_age.item(), img.size(0))
            LOSS2.update(loss_rank if isinstance(loss_rank, (int, float)) else loss_rank.item(), img.size(0))
            LOSS_CLS.update(loss_cls.item(), img.size(0))
            MAE.update(mae, img.size(0))
            DX_ACC.update(dx_acc, img.size(0))
            
            if i % opt.print_freq == 0:
                print(
                      'Epoch: [{0} / {1}]   [step {2}/{3}]\t'
                      'L_age {LOSS1.val:.3f} ({LOSS1.avg:.3f})\t'
                      'L_cls {LOSS_CLS.val:.3f} ({LOSS_CLS.avg:.3f})\t'
                      'L_rank {LOSS2.val:.3f} ({LOSS2.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {MAE.val:.3f} ({MAE.avg:.3f})\t'
                      'DX_Acc {DX_ACC.val:.3f} ({DX_ACC.avg:.3f})\t'.format
                      ( epoch, opt.epochs, i, len(train_loader)
                      , LOSS1=LOSS1, LOSS_CLS=LOSS_CLS, LOSS2=LOSS2, loss=losses, MAE=MAE, DX_ACC=DX_ACC ))
        else:
            if opt.model == 'ACDense':
                out = model(input, male)
            else:
                out = model(input)
                
            # =========== compute loss =========== #
            loss1 = criterion1(out, target)
            if opt.lbd > 0:
                loss2 = criterion2(out, target)
            else:
                loss2 = 0
            loss = loss1 + opt.lbd * loss2

            mae = metric(out.detach(), target.detach().cpu())
            losses.update(loss, img.size(0))
            LOSS1.update(loss1,img.size(0))
            LOSS2.update(loss2 if isinstance(loss2, (int, float)) else loss2.item(),img.size(0))
            MAE.update(mae, img.size(0))
            
            if i % opt.print_freq == 0:
                print(
                      'Epoch: [{0} / {1}]   [step {2}/{3}]\t'
                      'Loss1 {LOSS1.val:.3f} ({LOSS1.avg:.3f})\t'
                      'Loss2 {LOSS2.val:.3f} ({LOSS2.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {MAE.val:.3f} ({MAE.avg:.3f})\t'.format
                      ( epoch, opt.epochs, i, len(train_loader)
                      , LOSS1=LOSS1, LOSS2=LOSS2, loss=losses, MAE=MAE ))

        # =========== loss gradient back progation and optimizer parameter =========== #
        loss.backward()
        optimizer.step()

    if is_mtl:
        return losses.avg, MAE.avg, DX_ACC.avg, LOSS1.avg, LOSS_CLS.avg, LOSS2.avg
    else:
        return losses.avg, MAE.avg, LOSS1.avg, LOSS2.avg

def validate(valid_loader, model, criterion1, criterion2, device, criterion_ce=None, epoch=0):
    '''
    For validation process (supports both single-task and multi-task learning)
    
    Args:
        valid_loader (data loader): validation data loader.
        model (CNN model): convolutional neural network.
        criterion1 (loss fucntion): main loss function (age regression).
        criterion2 (loss fucntion): aux loss function (ranking).
        device (torch device type): default: GPU
        criterion_ce (loss function): classification loss for MTL mode
        epoch (int): current epoch for dynamic weight scheduling

    Returns:
        [float]: validation loss average, MAE average, and optionally classification accuracy
    '''
    losses = AverageMeter()
    MAE = AverageMeter()
    DX_ACC = AverageMeter()
    LOSS1 = AverageMeter()
    LOSS2 = AverageMeter()
    LOSS_CLS = AverageMeter()
    
    is_mtl = (criterion_ce is not None)
    
    # Use unified dynamic ranking loss weight function (same as training)
    if is_mtl:
        lbd_current = get_dynamic_ranking_weight(epoch, opt.lbd, warmup_start=10, warmup_end=20, max_lbd=2.0)
        cls_warmup = get_classification_warmup_factor(epoch, warmup_epochs=10)
    else:
        lbd_current = opt.lbd
        cls_warmup = 1.0

    # =========== switch to evaluate mode ===========#
    model.eval()

    with torch.no_grad():
        for _, batch in enumerate(valid_loader):
            if is_mtl:
                input, _, target, male, dx_label = batch
                dx_label = dx_label.to(device).long()
            else:
                input, _, target, male = batch
            
            # =========== prepare input data using standardized functions =========== #
            input = input.to(device)
            target = prepare_target(target, device)  # Unified target processing
            
            # Convert gender to one-hot encoding using torch.nn.functional.one_hot
            # Always prepare male input (even if use_gender=False) to avoid DataParallel issues
            male = prepare_gender_input(male, device)  # Unified gender one-hot encoding

            # =========== compute output and loss =========== #
            if is_mtl:
                # MTL forward: pass None for dx_true (no teacher forcing in validation) and epoch
                age_pred, cls_logits = model(input, male, None, epoch)
                
                # Age regression loss (heteroscedastic if enabled)
                if getattr(opt, 'age_hetero', False):
                    lv = getattr(model.module, 'age_logvar', None) if isinstance(model, nn.DataParallel) else getattr(model, 'age_logvar', None)
                    if lv is not None:
                        loss_age = 0.5 * torch.exp(-lv) * (age_pred - target)**2 + 0.5 * lv
                        loss_age = loss_age.mean()
                    else:
                        loss_age = criterion1(age_pred, target)
                else:
                    loss_age = criterion1(age_pred, target)
                
                # Classification loss with warmup (same as training)
                loss_cls_raw = criterion_ce(cls_logits, dx_label)
                loss_cls = cls_warmup * loss_cls_raw  # Apply warmup factor
                
                # Ranking loss (integrated into age task)
                if lbd_current > 0:
                    loss_rank = criterion2(age_pred, target)
                    loss_age_combined = loss_age + lbd_current * loss_rank
                else:
                    loss_rank = torch.tensor(0.0).to(device)
                    loss_age_combined = loss_age
                
                # Pairwise order loss（可选）
                if getattr(opt, 'pairwise_w', 0.0) > 0:
                    loss_pair = pairwise_order_loss(age_pred, target, margin=0.0, delta=getattr(opt, 'pair_delta', 2.0))
                    loss_age_combined = loss_age_combined + getattr(opt, 'pairwise_w', 0.0) * loss_pair
                else:
                    loss_pair = torch.tensor(0.0, device=device)
                
                # Uncertainty-weighted MTL loss with 2 tasks (age+ranking, classification)
                # Clamp log_vars to [-5, 5] to prevent negative total loss
                log_vars = model.module.log_vars if isinstance(model, nn.DataParallel) else model.log_vars
                log_vars_clamped = torch.clamp(log_vars, -5.0, 5.0)
                
                # Multi-task loss: L = exp(-s_i) * L_i + s_i
                loss = (torch.exp(-log_vars_clamped[0]) * loss_age_combined + log_vars_clamped[0] +
                        torch.exp(-log_vars_clamped[1]) * loss_cls + log_vars_clamped[1])
                
                # Compute metrics
                mae = metric(age_pred.detach(), target.detach().cpu())
                dx_acc = (cls_logits.argmax(1) == dx_label).float().mean().item()
                
                # =========== measure accuracy and record loss =========== #
                losses.update(loss.item(), input.size(0))
                MAE.update(mae, input.size(0))
                DX_ACC.update(dx_acc, input.size(0))
                LOSS1.update(loss_age.item(), input.size(0))
                LOSS_CLS.update(loss_cls.item(), input.size(0))
                LOSS2.update(loss_rank if isinstance(loss_rank, (int, float)) else loss_rank.item(), input.size(0))
            else:
                if opt.model == 'ACDense':
                    out = model(input, male)
                else:
                    out = model(input)
                # =========== compute loss =========== #
                loss1 = criterion1(out, target)
                if opt.lbd > 0:
                    loss2 = criterion2(out, target)
                else:
                    loss2 = 0
                loss = loss1 + opt.lbd * loss2
                mae = metric(out.detach(), target.detach().cpu())

                # =========== measure accuracy and record loss =========== #
                losses.update(loss, input.size(0))
                MAE.update(mae, input.size(0))
                LOSS1.update(loss1, input.size(0))
                LOSS2.update(loss2 if isinstance(loss2, (int, float)) else loss2.item(), input.size(0))
        
        if is_mtl:
            print(
                    'Valid: [steps {0}], Loss {loss.avg:.4f},  MAE: {MAE.avg:.4f},  DX_Acc: {DX_ACC.avg:.4f}'.format(
                    len(valid_loader), loss=losses, MAE=MAE, DX_ACC=DX_ACC))
            return losses.avg, MAE.avg, DX_ACC.avg, LOSS1.avg, LOSS_CLS.avg, LOSS2.avg
        else:
            print(
                    'Valid: [steps {0}], Loss {loss.avg:.4f},  MAE:  {MAE.avg:.4f}'.format(
                    len(valid_loader), loss=losses, MAE=MAE))
            return losses.avg, MAE.avg, LOSS1.avg, LOSS2.avg

def metric(output, target):
    target = target.data.numpy()
    pred = output.cpu()  
    pred = pred.data.numpy()
    mae = mean_absolute_error(target,pred)
    return mae

def save_checkpoint(state, is_best, out_dir, model_name):
    checkpoint_path = out_dir+model_name+'_checkpoint.pth.tar'
    best_model_path = out_dir+model_name+'_best_model.pth.tar'
    torch.save(state, checkpoint_path)
    if is_best:
        torch.save(state, best_model_path)
        print("=======>   This is the best model !!! It has been saved!!!!!!\n\n")

def weights_init(w):
    classname = w.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(w, 'weight'):
            # nn.init.kaiming_normal_(w.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(w.weight, mode='fan_in', nonlinearity='leaky_relu')
        if hasattr(w, 'bias') and w.bias is not None:
                nn.init.constant_(w.bias, 0)
    if classname.find('Linear') != -1:
        if hasattr(w, 'weight'):
            torch.nn.init.xavier_normal_(w.weight)
        if hasattr(w, 'bias') and w.bias is not None:
            nn.init.constant_(w.bias, 0)
    if classname.find('BatchNorm') != -1:
        if hasattr(w, 'weight') and w.weight is not None:
            nn.init.constant_(w.weight, 1)
        if hasattr(w, 'bias') and w.bias is not None:
            nn.init.constant_(w.bias, 0)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EarlyStopping:

    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=15, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 15
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_metric):

        score = val_metric

        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}\n\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0



if __name__ == "__main__":
    res = os.path.join(opt.output_dir, 'result.txt')
    
    # HPO mode: skip interactive prompt
    if opt.trial_id:
        print(f'=> HPO mode: trial_id={opt.trial_id}')
        if not os.path.exists(opt.output_dir):
            os.makedirs(opt.output_dir)
    else:
        # Normal mode: interactive prompt
        if os.path.isdir(opt.output_dir): 
            if input("### output_dir exists, rm? ###") == 'y':
                os.system('rm -rf {}'.format(opt.output_dir))

        # =========== set train folder =========== #
        if not os.path.exists(opt.output_dir):
            os.makedirs(opt.output_dir)
    
    print('=> training from scratch.\n')
    if os.path.exists(res):
        os.system('echo "train {}" >> {}'.format(datetime.datetime.now(), res))
    else:
        with open(res, 'w') as f:
            f.write(f'train {datetime.datetime.now()}\n')

    result = main(res)
    
    # Print final results
    if result:
        print('\n========== Training Complete ==========')
        print(f"Best Validation MAE: {result['best_mae']:.4f} (epoch {result['best_epoch']})")
        if result['best_acc'] is not None:
            print(f"Best Validation Acc: {result['best_acc']:.4f}")
        print(f"Test MAE: {result['test_mae']:.4f}")
        if result['test_acc'] is not None:
            print(f"Test Acc: {result['test_acc']:.4f}")
        print('=======================================')
