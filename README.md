# ParaOpt
## ParaOpt.py 

普通代码

串行：
    
    python ParaOpt.py --model_name cnn --training_mode serial --optimizer_type sgd --ema_decay 0.85 --threshold 1e-6 --max_step 60000

模拟并行：

    python ParaOpt.py --model_name cnn --training_mode parallel --optimizer_type sgd --ema_decay 0.85 --threshold 1e-6 --max_step 60000

## ParaOpt_mul.py

实时并行代码

    python ParaOpt_mul.py

    device_count 选择GPU数量，下面的程序同理

## ParaOpt_mul_Sweep.py

批量参数代码

批量实验，创建新的sweep和agent：

    python ParaOpt_mul_Sweep.py --sweep --agent --count 100

在已有的sweep项目中创建新的agent：
    
    python ParaOpt_mul_Sweep.py --sweep_id xxxx --agent --count 100

## Parallm_Sweep.py

大模型实时并行代码

运行单次：

    串行：
    python Parallm_Sweep.py --training_mode serial

    并行：
    python Parallm_Sweep.py --training_mode parallel

批量实验，创建新的sweep和agent：

    python Parallm_Sweep.py --sweep --agent --count 100

在已有的sweep项目中创建新的agent：
    
    python ParallmSweep.py --sweep_id xxxx --agent --count 100


