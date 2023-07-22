# Plasma Shape Control under the equilibrium of GS equation via Deep Reinforcement Learning
## Introduction
<p>
This is github repository for research on Tokamak plasma operation control based on deep reinforcement learning. We aim to control more than two targets with sustaining stable operation. In this research, we implement OpenAI gym based environment which contains neural networks that predict the state of the tokamak plasma, indicating that we can use GPU based computations for estimating the tokamak plasma profiles. 
</p>
<div>
    <p float = 'left'>
        <img src="/image/control_performance_new.gif"  width="720" height="320">
    </p>
</div>
<p>
Based on our approach, we can show some control performance under the virtual KSTAR environment resulted from neural networks. The poloidal flux of the plasma is estimated by Grad-Shafranov solver based on Physics-Informed Neural Network. For multi-target control, we will proceed development of the multi-objectve reinforcement learning algorithm for multi-objective tokamak plasma operation control. 
</p>

<div>
    <p float = 'left'>
        <img src="/image/example_prediction.png"  width="360" height="320">
        <img src="/image/example_reward.png"  width="360" height="320">
    </p>
</div>

<div>
    <p float = 'left'>
        <img src="/image/example_shape_control.png"  width="720" height="300">
    </p>
</div>

The summary of our proceeding work is listed as below.
- Targets
    - Plasma Beta-N | q95
    - Plasma boundary shape
- Neural networks for implementing virtual KSTAR tokamak environment
    - Prediction for plasma state: Non-stationary Transformer, Transformer, SciNet
    - Determination of the plasma poloidal flux: Physics-Informed Neural Network (Free-boundary Grad-Shafranov Solver)
- Reinforcement Learning algorithm for control
    - DDPG : Deterministic Deep Policy Gradient Method
    - SAC : Soft Actor Critic Algorithm
    - CAPS : Conditioning for Action Policy Smoothness (Smooth control)

## Scheme
Below is the overall scheme for our research. Our plan is to control both 0D parameters and shape parameters as a multi-objective control. Our final aim is to develop the framework for virtual tokamak plasma operation control under the V-DEMO project. The ML pipeline construction and integrated database for training RL based controller and prediction model will be the main purpose for future research. 

<div>
    <p float = 'left'>
        <img src="/image/scheme.png"  width="560" height="280">
    </p>
</div>

## How to Run
- Training 0D parameter predictor : Training code for NN-based predictor for RL environment
    ```
    python3 train_0D_predictor.py   --gpu_num {gpu number} --num_epoch {number of epoch}
                                    --batch_size {batch size} --lr {learning rate}
                                    --verbose {verbose} --gamma {decay constant}
                                    --max_norm_grad {maximum norm grad} --root_dir {directory for saving weights}
                                    --tag {tag name} --use_scaler {bool}
                                    --scaler {'Robust','Standard','MinMax'} --seq_len {input sequence length}
                                    --pred_len {output sequence length} --interval {data split interval}
                                    --model {algorithms for use : Transformer, NStransformer, SCINet}
                                    --use_forgetting {differentiate forgetting mechanism}
    ```
- Training shape predictor : Training code for Physics-informed Neural Network for GS solver
    ```
    python3 train_shape_predictor.py    --gpu_num {gpu number} --num_epoch {number of epoch}
                                        --batch_size {batch size} --lr {learning rate}
                                        --verbose {verbose} --gamma {decay constant}
                                        --max_norm_grad {maximum norm grad} --save_dir {directory for saving result}
                                        --tag {tag name} --GS_loss {weight for Grad-Shafranov loss}
                                        --Constraint_loss {weight for constraint loss for training GS solver}
    ```
- Training policy network : Training code for RL-based controller based on virtual KSTAR environment
    ```
    python3 train_controller.py --gpu_num {gpu number} --num_epoch {number of epoch}
                                --shot_random {bool} --t_init {initial time}
                                --t_terminal {terminal time} --dt {time interval}
                                --tag {tag name} --save_dir {directory for saving result}
                                --batch_size {batch size} --num_episode {number of episode}
                                --lr {learning rate} --gamma {decay constant}
                                --min_value {minimum value for action} --max_value {maximum value for action}
                                --tau {constant for updating policy} --verbose {verbose}
                                --predictor_weight {directory of saved weight file} --seq_len {input sequence length}
                                --pred_len {output sequence length} --algorithm {RL algorithm for use : SAC, DDPG}
                                --objective {objective for RL : params-control, shape-control, multi-objective}
    ```
- Playing policy network : Execute the autonomous control under the virtual KSTAR environment using the policy network
    ```
    python3 play_controller.py  --gpu_num {gpu number}
                                --shot_random {bool} --t_init {initial time}
                                --t_terminal {terminal time} --dt {time interval}
                                --tag {tag name} --save_dir {directory for saving result}
                                --predictor_weight {directory of saved weight file} --seq_len {input sequence length}
                                --pred_len {output sequence length} --algorithm {RL algorithm for use : SAC, DDPG}
                                --objective {objective for RL : params-control, shape-control, multi-objective}
    ```

## Additional result
In this part, we will share some result for virtual KSTAR tokamak operation control. The below picture describes the result of the training procedure with respect to the reward for each episode. DDPG and SAC are both applied and we found that the SAC shows possibility of smooth and robust control on virtual KSTAR tokamak operation.

<div>
    <p float = 'left'>
        <img src="/image/SAC_episode_reward.png"  width="640" height="400">
    </p>
</div>

And we play SAC algorithm with initial condition for KSTAR shot # 21747. We can observe the improved operation under the virtual KSTAR environment via RL control. The below picture is an example of the result for the purpose of 0D parameters control.
<div>
    <p float = 'left'>
        <img src="/image/SAC_shot_21747_operation_0D.png"  width="640" height="320">
    </p>
    <p float = 'left'>
        <img src="/image/SAC_shot_21747_operation_control.png"  width="640" height="400">
    </p>
</div>

## Details
### model / algorithm
- SCINet : https://arxiv.org/abs/2106.09305
- Transformer : https://arxiv.org/abs/1706.03762
- Non-stationary transformer : https://arxiv.org/abs/2205.14415
- DDPG : https://arxiv.org/abs/1509.02971
- SAC : https://arxiv.org/abs/1812.05905

### Dataset
- iKSTAR
- over 4000 experiments are used

## Reference
- Deep Neural Network-Based Surrogate Model for Optimal Component Sizing of Power Converters Using Deep Reinforcement Learning(https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9841555)
- Feedforward beta control in the KSTAR tokamak by deep reinforcement learning, Jaemin Seo et al : https://iopscience.iop.org/article/10.1088/1741-4326/ac121b/meta
- Magnetic control of tokamak plasmas through deep reinforcement learning, Jonas Degrave, Federico Felici et al : https://iopscience.iop.org/article/10.1088/1741-4326/ac121b/meta
- Development of an operation trajectory design algorithm for control of multiple 0D parameters using deep reinforcement learning, Jaemin Seo et al : https://iopscience.iop.org/article/10.1088/1741-4326/ac79be/meta 