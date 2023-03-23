# Plasma Shape Control under the equilibrium of GS equation via Deep Reinforcement Learning
## Introduction
- This is github repository for research on tokamak plasma operation control using Deep Reinforcement Learning
- We aim to control two targets with sustaining stable operation
(1) Beta-N over 3.0
(2) Plasma boundary shape control
- We replace tokamak environment by neural network based virtual environment. Transformer model is used as a 0D parameter predictor
- OpenAI gym based environment is implemented for training policy network. 
- We select DDPG as a policy gradient based RL due to its 2 significant benefits
(1) Efficient data sampling 
(2) Work well(?) on continuous action space
- The test shot control with initial information of shot 21747 in KSTAR

<div>
    <p float = 'left'>
        <img src="/image/shot_21474_operation_0D.png"  width="360" height="320">
        <img src="/image/shot_21474_operation_control.png"  width="360" height="320">
    </p>
</div>

## Scheme for this research
- Below is the overall scheme for our research
<div>
    <p float = 'left'>
        <img src="/image/scheme.png"  width="640" height="400">
    </p>
</div>

## How to Run
- Training 0D parameter predictor : Transformer based network
    ```
    python3 train_transformer.py --gpu_num {gpu number} --num_epoch {number of epoch}
                                 --batch_size {batch size} --lr {learning rate}
                                 --verbose {verbose} --gamma {decay constant}
                                 --max_norm_grad {maximum norm grad} --root_dir {directory for saving result}
                                 --tag {tag name} --use_scaler {bool}
                                 --scaler {'Robust','Standard','MinMax'} --seq_len {input sequence length}
                                 --pred_len {output sequence length} --interval {data split interval}
    ```
- Training 0D parameter predictor : SCINet based network
    ```
    python3 train_scinet.py --gpu_num {gpu number} --num_epoch {number of epoch}
                            --batch_size {batch size} --lr {learning rate}
                            --verbose {verbose} --gamma {decay constant}
                            --max_norm_grad {maximum norm grad} --root_dir {directory for saving result}
                            --tag {tag name} --use_scaler {bool}
                            --scaler {'Robust','Standard','MinMax'} --seq_len {input sequence length}
                            --pred_len {output sequence length} --interval {data split interval}
    ```
- Training policy network : DDPG algorithm
    ```
    python3 train_ddpg.py --gpu_num {gpu number} --num_epoch {number of epoch}
                          --shot_random {bool} --t_init {initial time}
                          --t_terminal {terminal time} --dt {time interval}
                          --tag {tag name} --save_dir {directory for saving result}
                          --batch_size {batch size} --num_episode {number of episode}
                          --lr {learning rate} --gamma {decay constant}
                          --min_value {minimum value for action} --max_value {maximum value for action}
                          --tau {constant for updating policy} --verbose {verbose}
                          --predictor_weight {directory of saved weight file} --seq_len {input sequence length}
                          --pred_len {output sequence length}
    ```
## Result
- The best performance of the virtual tokamak operation is obtained from SAC algorithm.
- The below picture describes the result of the training procedure with respect to the reward for each episode.
<div>
    <p float = 'left'>
        <img src="/image/SAC_episode_reward.png"  width="640" height="400">
    </p>
</div>

- And we play SAC algorithm with initial condition for KSTAR shot # 21747.
- We can observe the improved operation under the virtual KSTAR environment via RL control.
<div>
    <p float = 'left'>
        <img src="/image/SAC_shot_21747_operation_0D.png"  width="640" height="400">
    </p>
    <p float = 'left'>
        <img src="/image/SAC_shot_21747_operation_control.png"  width="640" height="400">
    </p>
</div>

## Detail
### model / algorithm
- SCINet : https://arxiv.org/abs/2106.09305
- Transformer : https://arxiv.org/abs/1706.03762
- DDPG : https://arxiv.org/abs/1509.02971

### Dataset
- iKSTAR 

## Reference
- Deep Neural Network-Based Surrogate Model for Optimal Component Sizing of Power Converters Using Deep Reinforcement Learning(https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9841555)
- Feedforward beta control in the KSTAR tokamak by deep reinforcement learning, Jaemin Seo et al : https://iopscience.iop.org/article/10.1088/1741-4326/ac121b/meta
- Magnetic control of tokamak plasmas through deep reinforcement learning, Jonas Degrave, Federico Felici et al : https://iopscience.iop.org/article/10.1088/1741-4326/ac121b/meta