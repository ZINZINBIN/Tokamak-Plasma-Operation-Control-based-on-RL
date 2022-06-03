import argparse
import torch
import gym
from pyvirtualdisplay import Display
from src.models.mpo import MPO

parser = argparse.ArgumentParser(description="training MPO algorithm")
parser.add_argument("--batch_size", type = int, default = 32)
parser.add_argument("--dual_constraint", type = float, default = 0.1)
parser.add_argument('--kl_mean_constraint', type=float, default=0.01)
parser.add_argument('--kl_var_constraint', type=float, default=0.0001)
parser.add_argument('--kl_constraint', type=float, default=0.01)
parser.add_argument('--discount_factor', type=float, default=0.99)
parser.add_argument('--alpha_mean_scale', type=float, default=1.0)
parser.add_argument('--alpha_var_scale', type=float, default=100.0)
parser.add_argument('--alpha_scale', type=float, default=10.0)
parser.add_argument('--alpha_mean_max', type=float, default=0.1)
parser.add_argument('--alpha_var_max', type=float, default=10.0)
parser.add_argument('--alpha_max', type=float, default=1.0)
parser.add_argument('--sample_episode_num', type=int, default=50)
parser.add_argument('--sample_episode_maxstep', type=int, default=300)
parser.add_argument('--sample_action_num', type=int, default=64)
parser.add_argument('--iteration_num', type=int, default=1024)
parser.add_argument('--episode_rerun_num', type=int, default=3)
parser.add_argument('--mstep_iteration_num', type=int, default=5)
parser.add_argument('--evaluate_period', type=int, default=10)
parser.add_argument('--evaluate_episode_num', type=int, default=100)
parser.add_argument('--evaluate_episode_maxstep', type=int, default=300)
parser.add_argument("--model_save_period", type = int, default = 8)
parser.add_argument("--lr", type = float, default = 3e-4)
parser.add_argument("--hidden_dim", type = int, default = 128)
parser.add_argument('--save_dir', type=str, default="./results/save_model/")
parser.add_argument('--render',type = bool, default = True)
parser.add_argument('--load', type=str, default=None)

args = parser.parse_args()

if __name__ == "__main__":

    display = Display(visible = False, size = (400,300))
    display.start()

    env = gym.make("LunarLander-v2").unwrapped
    env.reset()

    # device allocation(GPU)
    if torch.cuda.is_available():
        print("cuda available : ", torch.cuda.is_available())
        print("cuda device count : ", torch.cuda.device_count())
        device = "cuda:0"
    else:
        device = "cpu" 

    # training process
    print("\n")
    print("# training process : MPO algorithm")
    print("\n")
    
    agent = MPO(
        env = env,
        device = device,
        dual_constraint=args.dual_constraint,
        kl_mean_constraint=args.kl_mean_constraint,
        kl_var_constraint=args.kl_var_constraint,
        discount_factor=args.discount_factor,
        alpha_mean_scale = args.alpha_mean_scale,
        alpha_var_scale = args.alpha_var_scale, 
        alpha_scale = args.alpha_scale, 
        alpha_mean_max = args.alpha_mean_max, 
        alpha_var_max =args.alpha_var_max, 
        alpha_max = args.alpha_max, 
        sample_episode_num = args.sample_episode_num, 
        sample_episode_maxstep = args.sample_episode_maxstep, 
        sample_action_num = args.sample_action_num, 
        batch_size = args.batch_size, 
        episode_rerun_num = args.episode_rerun_num, 
        mstep_iteration_num = args.mstep_iteration_num, 
        evaluate_period = args.evaluate_period, 
        evaluate_episode_num = args.evaluate_episode_num, 
        evaluate_episode_maxstep = args.evaluate_episode_maxstep, 
        hidden_dim = args.hidden_dim, 
        lr = args.lr
        )

    agent.train(
        iteration_num = args.iteration_num,
        model_save_period=args.model_save_period,
        render = args.render,
        save_dir = args.save_dir
    )
    
    print("training MPO done .... !")
    print("\n")
    print("# evaluation process: MPO algorithm")
    evaluate_score = agent.evaluate()
    print("evaluate score(avg) : ", evaluate_score)

    env.close()