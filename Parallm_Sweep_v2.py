import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.multiprocessing as mp
from tqdm import tqdm
import argparse
import time
from datetime import timedelta
import numpy as np
import copy
import os
import sys
from sklearn.metrics import accuracy_score, f1_score
import wandb

class Config:
    # 训练参数
    batch_size = 8
    num_epochs = 1
    max_steps = 1000  # 总迭代步数
    learning_rate = 6e-5
    momentum = 0.9
    
    # 加速配置
    P = 7  # 窗口大小
    threshold = 7e-2  # 误差阈值
    ema_decay = 0.9  # 阈值指数移动平均衰减率
    adaptivity_type = 'mean'  # 自适应策略: 'mean' 或 'median'
    val_check_interval = 5  # 验证间隔(秒)
    visualize_progress = True  # 是否可视化进度
    display_time = True  # 是否显示运行时间
    
    # 系统配置
    seed = 42  # 随机种子
    device_count = torch.cuda.device_count()  # GPU数量
    max_length = 512  # 最大序列长度
    
    # 优化器类型
    optimizer_type = 'sgd'
    
    # 训练模式
    training_mode = 'parallel'

    def update_from_args(self, args):
        for key, value in vars(args).items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}

def calculate_metrics(preds, labels, ignore_index=-100):
    """计算准确率和F1分数"""
    # 展平预测和标签
    preds = preds.cpu().flatten()
    labels = labels.cpu().flatten()
    
    # 忽略填充标记
    mask = labels != ignore_index
    preds = preds[mask]
    labels = labels[mask]
    
    if len(labels) == 0:
        return 0.0, 0.0
    
    # 计算准确率
    accuracy = accuracy_score(labels, preds)
    
    # 计算F1分数(宏平均)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    
    return accuracy, f1

# 封装GPT-2模型
class GPT2Wrapper(nn.Module):
    def __init__(self):
        super(GPT2Wrapper, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.config.use_cache = False  # 禁用缓存以支持并行训练
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
    def clone(self, device, other=None):
        if other is None:
            other = GPT2Wrapper().to(device)
            
        with torch.no_grad():
            for param_to, param_from in zip(other.parameters(), self.parameters()):
                param_to.data = param_from.data.clone()
                
        return other
        
    def get_params(self):
        return list(self.parameters())

    def set_params(self, params, clone=True):
        my_params = self.get_params()
        for p, q in zip(my_params, params):
            if clone:
                p.data = q.data.clone().to(p.device)
            else:
                p.data = q.to(p.device)
                
    def set_grads_from_grads(self, grads):
        my_params = self.get_params()
        for p, grad in zip(my_params, grads):
            if grad is not None:
                p.grad = grad.to(p.device)
                
    def compute_error_from_model(self, other):
        my_params = self.get_params()
        other_params = other.get_params()
        
        with torch.no_grad():
            error = 0.0
            total_num = 0
            for p, q in zip(my_params, other_params):
                error += torch.linalg.norm(p-q).pow(2).item()
                total_num += np.prod(list(q.shape))
        return error / total_num * 1e6

def optimizer_state_clone(optimizer_from, optimizer_to):
    optimizer_to.load_state_dict(optimizer_from.state_dict())


def run(rank, total_ranks, queues, config, model, tokenizer, train_loader, test_loader,wandb_run):
    device = torch.device(f"cuda:{rank}")
    model = model.to(device)
    print('Start process', rank)

    if rank == 0:
        
        train_loop(config, model, tokenizer, queues, test_loader, device, wandb_run)
        for _ in range(total_ranks - 1):
            queues[0].put(None)
    else:
        # 初始化优化器
        if config.optimizer_type.lower() == 'sgd':
            worker_optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
        elif config.optimizer_type.lower() == 'adam':
            worker_optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        else:
            worker_optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
            
        run_worker(model, worker_optimizer, train_loader, queues, device, config.seed)

def run_worker(model, optimizer, dataloader, queues, device, seed_offset):
    data_iter = iter(dataloader)
    while True:
        ret = queues[0].get()
        if ret is None:
            return
            
        params, step = ret
        model.set_params(params, clone=False)
        
        res = take_step(model, optimizer, dataloader, data_iter, device, step, seed_offset)
        
        # 收集计算的梯度
        my_params = model.get_params()
        grads = [param.grad for param in my_params]
        
        # 将梯度和指标发送回主进程
        queues[1].put((grads, step, {'loss': res['loss'], 
                                    'perplexity': res['perplexity'],
                                    'accuracy': res['accuracy'],
                                    'f1_score': res['f1_score'],
                                    'batch_size': res['batch_size']}))

def take_step(model, optimizer, dataloader, data_iter, device, step, seed_offset):
    """执行单个训练步骤并返回梯度"""
    # 设置随机种子以确保可重现性
    np.random.seed(step + seed_offset)
    torch.manual_seed(step + seed_offset)
    torch.cuda.manual_seed(step + seed_offset)
    
    # 获取批次数据
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        batch = next(data_iter)

    batch = to_device(batch, device)

    # 确保有labels字段
    if 'labels' not in batch:
        batch['labels'] = batch['input_ids'].clone()    
    
    # 前向和后向传播
    optimizer.zero_grad()
    outputs = model(**batch)
    
    # 检查是否有损失值
    if outputs.loss is None:
        raise ValueError("Model did not return loss value. Check input and model configuration.")
    
    loss = outputs.loss
    loss.backward()
    
    # 计算困惑度
    perplexity = torch.exp(loss.cpu())
    
    # 计算准确率和F1分数
    with torch.no_grad():
        logits = outputs.logits.cpu()
        preds = torch.argmax(logits, dim=-1)
        accuracy, f1 = calculate_metrics(preds, batch['labels'])
    
    return {
        'loss': loss.item(),
        'perplexity': perplexity.item(),
        'accuracy': accuracy,
        'f1_score': f1,
        'batch_size': batch['input_ids'].size(0)
    }

def train_loop(config, model, tokenizer, queues, test_loader, device,wandb_run):
    T = config.max_steps
    P = min(config.P, T)  # 调整窗口大小不超过GPU数量
    thresh = config.threshold
    
    # 初始化模型和优化器数组
    models = [None for _ in range(T+1)]
    optimizers = [None for _ in range(T+1)]
    
    # 设置初始窗口
    begin_idx, end_idx = 0, P
    total_iters = 0
    
    # 初始化统计信息
    running_loss = 0.0
    running_perplexity = 0.0
    running_accuracy = 0.0
    running_f1 = 0.0
    running_total = 0

    
    # 克隆初始模型到窗口中的每个位置
    for step in range(P+1):
        models[step] = model.clone(device)
        
        # 选择优化器
        if config.optimizer_type.lower() == 'sgd':
            optimizers[step] = optim.SGD(models[step].parameters(), lr=config.learning_rate, momentum=config.momentum)
        elif config.optimizer_type.lower() == 'adam':
            optimizers[step] = optim.Adam(models[step].parameters(), lr=config.learning_rate)
        else:  # adamw default
            optimizers[step] = optim.AdamW(models[step].parameters(), lr=config.learning_rate)
    
    step_count = 0
    
    # 进度条初始化
    start_time = time.time()
    last_vis_time = 0
    pbar = tqdm(total=T)
    
    while begin_idx < T:
        # 计算窗口大小
        parallel_len = end_idx - begin_idx
        
        # 存储梯度预测
        pred_f = [None for _ in range(parallel_len)]
        metrics = [None for _ in range(parallel_len)]
        
        # 分发任务到工作进程
        for i in range(parallel_len):
            step = begin_idx + i
            params = [p.data for p in models[step].get_params()]
            queues[0].put((params, step))
        
        # 收集工作进程的梯度
        for i in range(parallel_len):
            _grads, _step, _metrics = queues[1].get()
            _i = _step - begin_idx
            pred_f[_i] = _grads
            metrics[_i] = _metrics
        
        # 从窗口起点开始执行定点迭代
        rollout_model = models[begin_idx]
        rollout_optimizer = optimizers[begin_idx]
        
        ind = None  # 重新同步点
        errors_all = 0  # 累积误差
        
        # 对窗口中的每个位置执行定点迭代
        for i in range(parallel_len):
            step = begin_idx + i
            
            # 设置之前计算的梯度并执行优化步骤
            rollout_model.set_grads_from_grads(pred_f[i])
            rollout_optimizer.step()
            rollout_optimizer.zero_grad()
            
            # 计算与预生成模型的误差
            error = rollout_model.compute_error_from_model(models[step+1])
            
            # 基于适应性类型计算总误差
            if config.adaptivity_type == 'median':
                if i == parallel_len // 2:
                    errors_all = error
            elif config.adaptivity_type == 'mean':
                errors_all += error / parallel_len
            
            # 如果误差超过阈值或到达窗口末尾，标记同步点
            if ind is None and (error > thresh or i == parallel_len - 1):
                ind = step + 1
                optimizer_state_clone(rollout_optimizer, optimizers[step+1])

            # 更新统计数据
            if ind is None or step < ind:
                step_count += 1
                _metrics = metrics[i]
                running_loss += _metrics['loss']
                running_perplexity += _metrics['perplexity']
                running_accuracy += _metrics['accuracy']
                running_f1 += _metrics['f1_score']
                running_total += _metrics['batch_size']
                
            # 从同步点开始克隆模型
            if ind is not None:
                models[step+1] = rollout_model.clone(device, models[step+1])
        
        # 更新阈值
        thresh = thresh * config.ema_decay + errors_all * (1 - config.ema_decay)
        
        # 滑动窗口
        new_begin_idx = ind
        new_end_idx = min(new_begin_idx + parallel_len, T)
        
        # 为新窗口区域克隆模型
        for step in range(end_idx+1, new_end_idx+1):
            models[step] = rollout_model.clone(device, 
                                              models[step - 1 - parallel_len] if step - 1 - parallel_len >= 0 else None)
            optimizers[step] = optimizers[step - 1 - parallel_len] if step - 1 - parallel_len >= 0 else None
        
        # 更新窗口位置和进度
        progress = new_begin_idx - begin_idx
        begin_idx = new_begin_idx
        end_idx = new_end_idx
        
        total_iters += 1
        pbar.update(progress)
        
        # 周期性输出进度
        if total_iters % 5 == 0 and begin_idx > 0:
            avg_loss = running_loss / begin_idx
            avg_perplexity = running_perplexity / begin_idx
            avg_accuracy = running_accuracy / begin_idx
            avg_f1 = running_f1 / begin_idx
            elapsed = time.time() - start_time
            elapsed_str = str(timedelta(seconds=int(elapsed))).split('.')[0]
            
            pbar.set_description(
                f'Loss: {avg_loss:.4f} | PPL: {avg_perplexity:.2f} | Acc: {avg_accuracy:.2%} | F1: {avg_f1:.2%} | Time: {elapsed_str}'
            )
            
            # Log metrics to wandb
            wandb_run.log({
                "Iter": total_iters,
                "Train_Loss": avg_loss,
                "Train_Acc": avg_accuracy,
                "Train_Perplexity": avg_perplexity,
                "Train_F1": avg_f1,
                "Original_Steps": begin_idx
            })
    
    # 训练结束
    pbar.close()
    
    # 最终测试
    final_model = models[T]
    del models[:T]
    del optimizers
    
    elapsed = time.time() - start_time
    
    test_loss, test_perplexity, test_accuracy, test_f1 = evaluate_model(final_model, test_loader, device)
    
    # Log final metrics to wandb
    wandb_run.log({
        "final_test_loss": test_loss,
        "final_test_perplexity": test_perplexity,
        "final_test_accuracy": test_accuracy,
        "final_test_f1": test_f1,
        "time":elapsed
    })
    
 
    elapsed_str = str(timedelta(seconds=int(elapsed))).split('.')[0]
    print(f"\nTraining completed in {elapsed_str}")
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test perplexity: {test_perplexity:.2f}")
    print(f"Final test accuracy: {test_accuracy:.2%}")
    print(f"Final test F1 score: {test_f1:.2%}")
    print(f"Total iterations: {total_iters} (vs {T} normal iterations)")
    print(f"Effective speed-up: {T/total_iters:.2f}x")
    
    return final_model

def train_loop_serial(config, model, tokenizer, train_loader, test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 选择优化器
    if config.optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    elif config.optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    else:  # adamw default
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # 训练循环
    start_time = time.time()
    running_loss = 0.0
    running_perplexity = 0.0
    running_accuracy = 0.0
    running_f1 = 0.0
    running_total = 0
    
    pbar = tqdm(total=config.max_steps)
    
    data_iter = iter(train_loader)
    for step in range(config.max_steps):
        # 获取数据
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        batch = to_device(batch, device)
        
        # 前向和后向传播
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        # 计算困惑度
        perplexity = torch.exp(loss)
        
        # 计算准确率和F1分数
        with torch.no_grad():
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            accuracy, f1 = calculate_metrics(preds, batch['labels'])
        
        # 更新统计数据
        running_loss += loss.item()
        running_perplexity += perplexity.item()
        running_accuracy += accuracy
        running_f1 += f1
        running_total += batch['input_ids'].size(0)
        
        # 更新进度条
        if (step + 1) % 5 == 0:
            avg_loss = running_loss / (step + 1)
            avg_perplexity = running_perplexity / (step + 1)
            avg_accuracy = running_accuracy / (step + 1)
            avg_f1 = running_f1 / (step + 1)
            elapsed = time.time() - start_time
            elapsed_str = str(timedelta(seconds=int(elapsed))).split('.')[0]
            
            pbar.set_description(
                f'Loss: {avg_loss:.4f} | PPL: {avg_perplexity:.2f} | Acc: {avg_accuracy:.2%} | F1: {avg_f1:.2%} | Time: {elapsed_str}'
            )
            
            # Log metrics to wandb
            wandb.log({
                "Iter": step + 1,
                "Train_Loss": avg_loss,
                "Train_Acc": avg_accuracy,
                "Train_Perplexity": avg_perplexity,
                "Train_F1": avg_f1,
                "Original_Steps": step + 1
            })
        
        pbar.update(1)
    
    pbar.close()
    
    elapsed = time.time() - start_time
        
    # 最终测试
    test_loss, test_perplexity, test_accuracy, test_f1 = evaluate_model(model, test_loader, device)

    # Log final metrics to wandb
    wandb.log({
        "final_test_loss": test_loss,
        "final_test_perplexity": test_perplexity,
        "final_test_accuracy": test_accuracy,
        "final_test_f1": test_f1,
        "time":elapsed
    })
    
    elapsed_str = str(timedelta(seconds=int(elapsed))).split('.')[0]
    print(f"\nTraining completed in {elapsed_str}")
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test perplexity: {test_perplexity:.2f}")
    print(f"Final test accuracy: {test_accuracy:.2%}")
    print(f"Final test F1 score: {test_f1:.2%}")
    
    return model

def evaluate_model(model, test_loader, device):
    """评估模型性能"""
    model.eval()
    total_loss = 0.0
    total_perplexity = 0.0
    total_accuracy = 0.0
    total_f1 = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            perplexity = torch.exp(loss)
            
            # 计算准确率和F1分数
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            accuracy, f1 = calculate_metrics(preds, batch['labels'])
            
            total_loss += loss.item()
            total_perplexity += perplexity.item()
            total_accuracy += accuracy
            total_f1 += f1
            total_batches += 1
    
    avg_loss = total_loss / total_batches
    avg_perplexity = total_perplexity / total_batches
    avg_accuracy = total_accuracy / total_batches
    avg_f1 = total_f1 / total_batches
    model.train()
    return avg_loss, avg_perplexity, avg_accuracy, avg_f1

def tokenize_function(examples, tokenizer, max_length):
    # 使用相同的文本作为输入和标签
    tokenized_inputs = tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=max_length,
        return_tensors="pt"
    )
    
    # 添加labels字段
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
    return tokenized_inputs

def setup_sweep_configuration():
    """设置wandb sweep配置"""
    sweep_config = {
            'method': 'random',  
            'metric': {
                'name': 'time',  
                'goal': 'minimize' 
            },
            'parameters': {
                'learning_rate': {
                    'distribution':'log_uniform_values',
                    'min': 1e-5,
                    'max': 1e-4                
                },
                'batch_size': {
                    'values': [8, 16]
                    # 'distribution':'q_uniform',
                    # 'q': 2,
                    # 'min': 8,
                    # 'max': 16,
                },
                'optimizer_type': {
                    'values': ['SGD', 'Adam', 'AdamW']
                },
                # 'P': {  
                #     # 'values': [5, 7, 10, 15]
                #     'distribution':'q_uniform',
                #     'q': 1,
                #     'min': 2,
                #     'max': 15,                
                # },
                'threshold': {              
                    'min': 1e-6,
                    'max': 1e-1
                },
                'ema_decay': {
                    'min': 0.01,
                    'max': 0.99
                }
                # 'adaptivity_type': {
                #     'values': ['mean', 'median']
                # },
                # 'model_name': {
                #     'values': ['cnn', 'resnet18', 'mobilenet_v2']
                # },
                # 'training_mode': {
                #     'values': ['parallel', 'serial']
                # }
            }
        }
    
    return sweep_config

def train_with_config(config_dict=None):
    """专为wandb.sweep设计的训练函数"""
    
    # 初始化基本配置
    config = Config()
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    
    # 初始化wandb
    wandb_run = wandb.init(project='NIPS_2025_ParaOptimizer', config=config_dict)
    
    # 使用wandb的config更新参数（允许sweep覆盖）
    for key, value in wandb.config.items():
        if hasattr(config, key):
            setattr(config, key, value)
            
    from datasets import disable_caching
    disable_caching()
    
    # 加载数据集和tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset("wikitext", "wikitext-2-v1",download_mode="reuse_dataset_if_exists")
    
    # 数据预处理
    def preprocess(examples):
        return tokenize_function(examples, tokenizer, config.max_length)
    
    tokenized_datasets = dataset.map(
        preprocess,
        batched=True,
        remove_columns=["text"],
    )
    
    # 设置数据格式为torch tensors
    tokenized_datasets.set_format("torch")
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        tokenized_datasets["train"],
        batch_size=config.batch_size,
        shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        tokenized_datasets["validation"],
        batch_size=config.batch_size,
        shuffle=False
    )
    
    # 创建模型
    model = GPT2Wrapper()
    
    # 选择训练模式
    if config.training_mode.lower() == 'parallel':
        torch.autograd.set_detect_anomaly(True)
        mp.set_start_method('spawn', force=True)
        queues = mp.Queue(), mp.Queue(), mp.Queue()

        processes = []
        num_processes = min(config.device_count, torch.cuda.device_count())

        if num_processes == 1:
            run(0, 1, queues, config, model, tokenizer, train_loader, test_loader,wandb_run)
        else:
            for rank in range(num_processes):
                p = mp.Process(target=run, args=(rank, num_processes, queues, config, model, tokenizer, train_loader, test_loader,wandb_run))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
    else:
        # 串行训练
        train_loop_serial(config, model, tokenizer, train_loader, test_loader)
    
    # 关闭wandb run
    wandb.finish()
    
    return model

def setup_arg_parser():
    parser = argparse.ArgumentParser(description='ParaOpt for Language Models with wandb sweep')

    parser.add_argument('--sweep', action='store_true', help='Run wandb sweep')
    parser.add_argument('--agent', action='store_true', help='Run as a sweep agent')
    parser.add_argument('--sweep_id', type=str, help='Sweep ID to use for agent')
    parser.add_argument('--count', type=int, default=10, help='Number of runs for the sweep')

    parser.add_argument('--device_count', type=int, help='Number of CUDA devices to use')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--max_steps', type=int, help='max steps')
    parser.add_argument('--learning_rate', type=float, help='learning rate')

    parser.add_argument('--P', type=int, help='window size')
    parser.add_argument('--threshold', type=float, help='threshold')
    parser.add_argument('--ema_decay', type=float, help='阈值指数移动平均衰减率')
    parser.add_argument('--adaptivity_type', type=str, choices=['mean', 'median'], help='the computation type of error_all')

    parser.add_argument('--optimizer_type', type=str, choices=['sgd', 'adam', 'adamw'], help='the type of optimizer')
    parser.add_argument('--training_mode', type=str, choices=['parallel', 'serial'], help='simulation_parallel or serial')
    
    parser.add_argument('--max_length', type=int, help='maximum sequence length')
    
    return parser

def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    if args.sweep:
        # 创建新的sweep
        from config.sweep_config_llm2 import sweep_config
        # sweep_config = setup_sweep_configuration()
        sweep_id = wandb.sweep(sweep_config, project='NIPS_2025_ParaOptimizer')
        print(f"Created sweep with ID: {sweep_id}")
        
        if args.agent:
            # 直接运行代理
            wandb.agent(sweep_id, function=train_with_config, count=args.count)
    
    elif args.agent and args.sweep_id:
        # 使用现有的sweep ID运行代理
        wandb.agent(args.sweep_id, function=train_with_config, project='NIPS_2025_ParaOptimizer', count=args.count)
    
    else:
        # 常规的单次运行，使用命令行参数
        config = Config()
        config.update_from_args(args)
        
        # 创建从args转换来的config_dict
        config_dict = {k: v for k, v in vars(args).items() if hasattr(config, k) and v is not None}
        train_with_config(config_dict)

if __name__ == "__main__":
    main()