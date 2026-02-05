#!/bin/bash

################################################################################
# DARE-Net Training Script
# 
# Train DARE-Net (Diagnosis-Aware Routing Mixture-of-Experts) for 
# Brain Age Estimation with Multi-Task Learning
#
# Paper: "DARE-Net: Diagnosis-Aware Routing Mixture-of-Experts for Accurate 
#         and Clinically Interpretable Brain Age Estimation"
# 
# Best hyperparameters are loaded from Best/hyperparameter.json
# 
# Usage: bash script/bash_train_darenet.sh
################################################################################

echo "======================================================================"
echo "DARE-Net: Diagnosis-Aware Routing Mixture-of-Experts"
echo "Brain Age Estimation with Multi-Task Learning"
echo "======================================================================"
echo ""

# ============= Check Best folder exists ============= #
BEST_DIR="./Best"
HYPERPARAM_FILE="${BEST_DIR}/hyperparameter.json"

if [ ! -f "$HYPERPARAM_FILE" ]; then
    echo "Error: Best hyperparameters file not found: $HYPERPARAM_FILE"
    echo "Please ensure Best folder contains hyperparameter.json"
    exit 1
fi

echo "Loading best hyperparameters from: $HYPERPARAM_FILE"
echo ""

# ============= Extract best hyperparameters using Python ============= #
read -r -d '' PYTHON_SCRIPT << 'EOF'
import json
import sys

with open(sys.argv[1], 'r') as f:
    config = json.load(f)

# Extract parameters with defaults
params = {
    'lr': config.get('lr', 1e-3),
    'weight_decay': config.get('weight_decay', 5e-4),
    'loss': config.get('loss', 'mse'),
    'aux_loss': config.get('aux_loss', 'ranking'),
    'lbd': config.get('lbd', 2.0),
    'beta': config.get('beta', 1.0),
    'moe_num_experts': config.get('moe_num_experts', 8),
    'moe_topk': config.get('moe_topk', 3),
    'moe_gate_temp': config.get('moe_gate_temp', 1.5),
    'moe_alpha': config.get('moe_alpha', 0.02),
    'moe_entropy_w': config.get('moe_entropy_w', 0.004),
    'moe_tf_epochs': config.get('moe_tf_epochs', 16),
    'schedular': config.get('schedular', 'cosine'),
    'warmup_epoch': config.get('warmup_epoch', 4),
    'warmup_lr_init': config.get('warmup_lr_init', 3e-6),
    'min_lr': config.get('min_lr', 1e-4),
    'batch_size': config.get('batch_size', 4),
    'epochs': config.get('epochs', 100),
    'num_workers': config.get('num_workers', 8),
    'use_gender': config.get('use_gender', True),
    'use_timm_sched': config.get('use_timm_sched', True),
    'lr_decay_epochs': config.get('lr_decay_epochs', 30),
    'lr_decay_rate': config.get('lr_decay_rate', 0.1),
    'pairwise_w': config.get('pairwise_w', 0.23),
    'pair_delta': config.get('pair_delta', 2.0),
    'age_hetero': config.get('age_hetero', True),
}

# Print as bash variables
for key, value in params.items():
    if isinstance(value, bool):
        if value:
            print(f"{key}=true")
        else:
            print(f"{key}=false")
    else:
        print(f"{key}={value}")
EOF

# Execute Python script to extract parameters
eval $(python3 -c "$PYTHON_SCRIPT" "$HYPERPARAM_FILE")

# ============= Set default paths (modify as needed) ============= #
# Data Paths
label=/root/autodl-tmp/data/wfz_OASIS/Dataset.csv
train_data=/root/autodl-tmp/data/wfz_OASIS/train
valid_data=/root/autodl-tmp/data/wfz_OASIS/val
test_data=/root/autodl-tmp/data/wfz_OASIS/test

# Sorter Path (adjust batch_size in path if needed)
sorter_path=./darenet/Sodeep_pretrain_weight/Tied_rank_best_lstmla_slen_${batch_size}.pth.tar

# Output Path
save_path=./output/DARENet_best/

# Model - DARE-Net uses ACDense backbone with Diagnosis-Aware MoE
model=DARENet

# Training Settings
dis_range=5
print_freq=40

echo "DARE-Net Best Hyperparameters:"
echo "======================================================================"
echo "Learning & Optimization:"
echo "  - Learning rate: $lr"
echo "  - Weight decay: $weight_decay"
echo "  - Scheduler: $schedular"
echo "  - Warmup epochs: $warmup_epoch"
echo "  - Warmup LR init: $warmup_lr_init"
echo "  - Min LR: $min_lr"
echo "  - Use timm scheduler: $use_timm_sched"
echo "  - LR decay epochs: $lr_decay_epochs"
echo "  - LR decay rate: $lr_decay_rate"
echo ""
echo "Loss Functions:"
echo "  - Main loss: $loss"
echo "  - Auxiliary loss: $aux_loss"
echo "  - Ranking weight (lbd): $lbd"
echo "  - Classification weight (beta): $beta"
echo "  - Pairwise order loss weight: $pairwise_w"
echo "  - Pair delta: $pair_delta"
echo "  - Heteroscedastic regression: $age_hetero"
echo ""
echo "Diagnosis-Aware Routing MoE:"
echo "  - Number of experts: $moe_num_experts"
echo "  - Top-K routing: $moe_topk"
echo "  - Gate temperature: $moe_gate_temp"
echo "  - Load balance weight (alpha): $moe_alpha"
echo "  - Entropy regularization: $moe_entropy_w"
echo "  - Scheduled teacher forcing epochs: $moe_tf_epochs"
echo ""
echo "Training Configuration:"
echo "  - Model: $model (ACDense backbone)"
echo "  - Batch size: $batch_size"
echo "  - Total epochs: $epochs"
echo "  - Num workers: $num_workers"
echo "  - Use gender: $use_gender"
echo "  - Output: $save_path"
echo ""
echo "======================================================================"
echo ""

# Auto-start training
echo "Starting DARE-Net training with best parameters..."
echo ""

# ============= Train DARE-Net Model ============= #
CUDA_VISIBLE_DEVICES=0 python ./darenet/train_first_stage.py \
    --batch_size $batch_size \
    --epochs $epochs \
    --lr $lr \
    --weight_decay $weight_decay \
    --loss $loss \
    --aux_loss $aux_loss \
    --lbd $lbd \
    --beta $beta \
    --train_folder ${train_data} \
    --valid_folder ${valid_data} \
    --test_folder ${test_data} \
    --excel_path ${label} \
    --model ${model} \
    --output_dir ${save_path} \
    --sorter ${sorter_path} \
    --num_workers $num_workers \
    --use_gender $use_gender \
    --use_moe \
    --moe_num_experts $moe_num_experts \
    --moe_topk $moe_topk \
    --moe_gate_temp $moe_gate_temp \
    --moe_alpha $moe_alpha \
    --moe_entropy_w $moe_entropy_w \
    --moe_use_dx \
    --moe_tf_epochs $moe_tf_epochs \
    --dis_range $dis_range \
    --print_freq $print_freq \
    --schedular $schedular \
    --warmup_epoch $warmup_epoch \
    --warmup_lr_init $warmup_lr_init \
    --min_lr $min_lr \
    $( [ "$use_timm_sched" = "true" ] && echo "--use_timm_sched" ) \
    --lr_decay_epochs $lr_decay_epochs \
    --lr_decay_rate $lr_decay_rate \
    $( [ "$pairwise_w" != "0.0" ] && [ "$pairwise_w" != "0" ] && echo "--pairwise_w $pairwise_w --pair_delta $pair_delta" ) \
    $( [ "$age_hetero" = "true" ] && echo "--age_hetero" )

# ============= Check Results ============= #
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "DARE-Net Training Complete!"
    echo "======================================================================"
    echo ""
    echo "Model saved to: $save_path"
    echo ""
    echo "Results:"
    echo "  - Best checkpoint: ${save_path}DARENet_best_model.pth.tar"
    echo "  - Detailed metrics: ${save_path}metrics_epoch.csv"
    echo "  - Training log: ${save_path}training_log.txt"
    echo "  - Hyperparameters: ${save_path}hyperparameter.json"
    echo "  - Final metrics: ${save_path}metrics.json"
    echo ""
    echo "Paper Reference:"
    echo "  DARE-Net: Diagnosis-Aware Routing Mixture-of-Experts for"
    echo "  Accurate and Clinically Interpretable Brain Age Estimation"
    echo "  Medical Image Analysis, 2026"
    echo ""
    echo "======================================================================"
    echo ""
else
    echo ""
    echo "======================================================================"
    echo "DARE-Net Training failed!"
    echo "======================================================================"
    echo "Check the error messages above."
    exit 1
fi

