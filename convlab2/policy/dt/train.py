# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 16:14:07 2019
@author: truthless
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
import pickle as p
from torch import multiprocessing as mp
from convlab2.dialog_agent.agent import PipelineAgent
from convlab2.dialog_agent.env import Environment
from convlab2.nlu.svm.multiwoz import SVMNLU
from convlab2.dst.rule.multiwoz import RuleDST
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.policy.rlmodule import Memory, Transition
from convlab2.nlg.template.multiwoz import TemplateNLG
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from argparse import ArgumentParser
from convlab2.policy.ppo import PPO

# sys.path.append('/scratch/ycw30/decision-transformer/atari')
from torch.utils.data import Dataset
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig
# from run_dt_atari import StateActionReturnDataset as SARDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    mp = mp.get_context('spawn')
except RuntimeError:
    pass

class StateActionReturnDataset(Dataset):

    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):        
        self.block_size = block_size
#         self.vocab_size = max(actions) + 1
        self.vocab_size = 209
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
#         print(idx, block_size)
        for i in self.done_idxs:
            if i > idx and i > done_idx: # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
#         print(idx, done_idx)
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        states = states / 255.
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long)#.unsqueeze(1) # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)
        
#         for i in states:
#         print(states.shape)
        return states, actions, rtgs, timesteps

def sampler(pid, queue, evt, env, policy, batchsz):
    """
    This is a sampler function, and it will be called by multiprocess.Process to sample data from environment by multiple
    processes.
    :param pid: process id
    :param queue: multiprocessing.Queue, to collect sampled data
    :param evt: multiprocessing.Event, to keep the process alive
    :param env: environment instance
    :param policy: policy network, to generate action from current policy
    :param batchsz: total sampled items
    :return:
    """
    buff = Memory()

    # we need to sample batchsz of (state, action, next_state, reward, mask)
    # each trajectory contains `trajectory_len` num of items, so we only need to sample
    # `batchsz//trajectory_len` num of trajectory totally
    # the final sampled number may be larger than batchsz.

    sampled_num = 0
    sampled_traj_num = 0
    traj_len = 50
    real_traj_len = 0

    while sampled_num < batchsz:
        # for each trajectory, we reset the env and get initial state
        s = env.reset()

        for t in range(traj_len):

            # [s_dim] => [a_dim]
            s_vec = torch.Tensor(policy.vector.state_vectorize(s))
            a = policy.predict(s)

            # interact with env
            next_s, r, done = env.step(a)

            # a flag indicates ending or not
            mask = 0 if done else 1

            # get reward compared to demostrations
            next_s_vec = torch.Tensor(policy.vector.state_vectorize(next_s))

            # save to queue
            buff.push(s_vec.numpy(), policy.vector.action_vectorize(a), r, next_s_vec.numpy(), mask)

            # update per step
            s = next_s
            real_traj_len = t

            if done:
                break

        # this is end of one trajectory
        sampled_num += real_traj_len
        sampled_traj_num += 1
        # t indicates the valid trajectory length

    # this is end of sampling all batchsz of items.
    # when sampling is over, push all buff data into queue
    queue.put([pid, buff])
    evt.wait()


def sample(env, policy, batchsz, process_num):
    """
    Given batchsz number of task, the batchsz will be splited equally to each processes
    and when processes return, it merge all data and return
	:param env:
	:param policy:
    :param batchsz:
	:param process_num:
    :return: batch
    """

    # batchsz will be splitted into each process,
    # final batchsz maybe larger than batchsz parameters
    process_batchsz = np.ceil(batchsz / process_num).astype(np.int32)
    # buffer to save all data
    queue = mp.Queue()

    # start processes for pid in range(1, processnum)
    # if processnum = 1, this part will be ignored.
    # when save tensor in Queue, the process should keep alive till Queue.get(),
    # please refer to : https://discuss.pytorch.org/t/using-torch-tensor-over-multiprocessing-queue-process-fails/2847
    # however still some problem on CUDA tensors on multiprocessing queue,
    # please refer to : https://discuss.pytorch.org/t/cuda-tensors-on-multiprocessing-queue/28626
    # so just transform tensors into numpy, then put them into queue.
    evt = mp.Event()
    processes = []
    for i in range(process_num):
        process_args = (i, queue, evt, env, policy, process_batchsz)
        processes.append(mp.Process(target=sampler, args=process_args))
    for p in processes:
        # set the process as daemon, and it will be killed once the main process is stoped.
        p.daemon = True
        p.start()

    # we need to get the first Memory object and then merge others Memory use its append function.
    pid0, buff0 = queue.get()
    for _ in range(1, process_num):
        pid, buff_ = queue.get()
        buff0.append(buff_)  # merge current Memory into buff0
    evt.set()

    # now buff saves all the sampled data
    buff = buff0

    return buff.get_batch()


def collect(env, policy, batchsz, process_num):
    # sample data asynchronously
    batch = sample(env, policy, batchsz, process_num)
    print ('Data collection is done')

    # data in batch is : batch.state: ([1, s_dim], [1, s_dim]...)
    # batch.action: ([1, a_dim], [1, a_dim]...)
    # batch.reward/ batch.mask: ([1], [1]...)
#     s = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
#     a = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
#     r = torch.from_numpy(np.stack(batch.reward)).to(device=DEVICE)
#     mask = torch.Tensor(np.stack(batch.mask)).to(device=DEVICE)
    s = np.stack(batch.state)
    a = np.stack(batch.action)
    r = np.stack(batch.reward)
    mask = np.stack(batch.mask)
    mask = np.where(mask == 0)[0]
#     batchsz_real = s.size(0)
    
    ret = r # fixme
    
    # -- create reward-to-go dataset
    start_index = 0
    rtg = np.zeros_like(ret)
    for i in mask:
        traj_r = ret[start_index:i+1] # includes i
        for j in range(i-1, start_index-1, -1): # start from i-1
            rtg_j = traj_r[j-start_index:i+1-start_index]
            rtg[j] = sum(rtg_j) # includes i
        start_index = i+1
    print('max rtg is %d' % max(rtg))

    # -- create timestep dataset
    start_index = 0
    timesteps = np.zeros(len(a)+1, dtype=int)
    for i in mask:
        timesteps[start_index:i+1] = np.arange(i+1 - start_index)
        start_index = i+1
    print('max timestep is %d' % max(timesteps))

    return s, a, mask, rtg, timesteps

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--load_path", type=str, default="", help="path of model to load")
    parser.add_argument("--batchsz", type=int, default=102, help="batch size of trajactory sampling")
    parser.add_argument("--epoch", type=int, default=200, help="number of epochs to train")
    parser.add_argument("--process_num", type=int, default=4, help="number of processes of trajactory sampling")
    parser.add_argument("--config", type=str, default='config.json', help="config file")
    parser.add_argument('--model_type', type=str, default='reward_conditioned')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--context_length', type=int, default=30)
    args = parser.parse_args()

    dst_usr = None
    dst_sys = RuleDST()
    policy_sys = PPO(True, config=args.config) # use MLE pretrained policy to collect data
    policy_sys.load(args.load_path)
    policy_usr = RulePolicy(character='usr')
    simulator = PipelineAgent(None, None, policy_usr, None, 'user')
    evaluator = MultiWozEvaluator()
    env = Environment(None, simulator, None, dst_sys, evaluator)
    
    args.context_length = 20 # ???
    s, a, mask, rtgs, timesteps = collect(env, policy_sys, args.batchsz, args.process_num)
    
    # ref:
    # https://github.com/kzl/decision-transformer/blob/master/atari/run_dt_atari.py
    # setup transformer and train
    
    train_dataset = StateActionReturnDataset(s, args.context_length*3, a, mask, rtgs, timesteps)
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, \
                      n_layer=6, n_head=8, n_embd=128, \
                      model_type=args.model_type,max_timestep=max(timesteps))
    model = GPT(mconf)
    tconf = TrainerConfig(max_epochs=args.epoch, batch_size=args.batch_size, learning_rate=6e-4, \
                          lr_decay=True, warmup_tokens=512*20, \
                          final_tokens=2*len(train_dataset)*args.context_length*3, \
                          num_workers=4, seed=args.seed, model_type=args.model_type, \
                          game='MultiWoZ', max_timestep=max(timesteps))
    trainer = Trainer(model, train_dataset, None, tconf, env, policy_sys)
    trainer.train()

