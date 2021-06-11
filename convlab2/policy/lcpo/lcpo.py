# -*- coding: utf-8 -*-
import torch
from torch import optim
import numpy as np
import logging
import os
import json
from convlab2.policy.policy import Policy
from convlab2.policy.rlmodule import MultiDiscretePolicy, Value
from convlab2.util.train_util import init_logging_handler
from convlab2.policy.vector.vector_multiwoz import MultiWozVector
from convlab2.util.file_util import cached_path
import zipfile
import sys
from sklearn.metrics.pairwise import cosine_similarity
import pickle as p
import torch.nn.functional as F

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LCPO(Policy):

    def __init__(self, is_train=False, dataset='Multiwoz', config='config.json'):

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config/'+config), 'r') as f:
            cfg = json.load(f)
        self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg['save_dir'])
        self.save_per_epoch = cfg['save_per_epoch']
        self.update_round = cfg['update_round']
        self.optim_batchsz = cfg['batchsz']
        self.gamma = cfg['gamma']
        self.epsilon = cfg['epsilon']
        self.tau = cfg['tau']
        self.is_train = is_train
        self.reward_scale = cfg['reward_scale']
        self.adv_est = cfg['adv_est']
        self.partial_reward = cfg['partial_reward']
        
        if is_train:
            init_logging_handler(os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg['log_dir']))

        # construct policy and value network
        if dataset == 'Multiwoz':
            voc_file = os.path.join(root_dir, 'data/multiwoz/sys_da_voc.txt')
            voc_opp_file = os.path.join(root_dir, 'data/multiwoz/usr_da_voc.txt')
            self.vector = MultiWozVector(voc_file, voc_opp_file)
            self.policy = MultiDiscretePolicy(self.vector.state_dim, cfg['h_dim'], self.vector.da_dim).to(device=DEVICE)
            

        self.value = Value(self.vector.state_dim, cfg['hv_dim']).to(device=DEVICE)
        if is_train:
            self.policy_optim = optim.RMSprop(self.policy.parameters(), lr=cfg['policy_lr'])
            self.value_optim = optim.Adam(self.value.parameters(), lr=cfg['value_lr'])
        
    def predict(self, state):
        """
        Predict an system action given state.
        Args:
            state (dict): Dialog state. Please refer to util/state.py
        Returns:
            action : System act, with the form of (act_type, {slot_name_1: value_1, slot_name_2, value_2, ...})
        """
        s_vec = torch.Tensor(self.vector.state_vectorize(state))
        a = self.policy.select_action(s_vec.to(device=DEVICE), False).cpu()
        action = self.vector.action_devectorize(a.numpy())
        state['system_action'] = action
        
#         p.dump([state,s_vec,action,a],open( "sa.pkl", "wb" ))
#         exit()
        return action

    def init_session(self):
        """
        Restore after one session
        """
        pass
    
    def est_adv(self, r, v, mask, idx):
        """
        we save a trajectory in continuous space and it reaches the ending of current trajectory when mask=0.
        :param r: reward, Tensor, [b]
        :param v: estimated value, Tensor, [b]
        :param mask: indicates ending for 0 otherwise 1, Tensor, [b]
        :return: A(s, a), V-target(s), both Tensor
        """
        batchsz = v.size(0)

        # v_target is worked out by Bellman equation.
        v_target = torch.Tensor(batchsz).to(device=DEVICE)
        v_pseudo = torch.Tensor(batchsz).to(device=DEVICE)
        delta = torch.Tensor(batchsz).to(device=DEVICE)
        A_sa = torch.Tensor(batchsz).to(device=DEVICE)
        
#         prev_v_target = 0
#         prev_v = 0
#         prev_A_sa = 0
        
        for t in range(batchsz):
            if mask[t]!=0:
                r[t] = -1.0
                
        r /= self.reward_scale
        
        for t in reversed(range(batchsz)):
            if mask[t]==0:
                prev_A_sa = 0
                if self.partial_reward:
                    prev_v_pseudo = prev_v_target = prev_v = v[t]
                else:
                    prev_v_pseudo = prev_v_target = prev_v = 0
            
            v_pseudo[t] = r[t] + self.gamma * prev_v_pseudo * mask[t]
            delta[t] = r[t] + self.gamma * prev_v * mask[t] - v[t]
            A_sa[t]  = delta[t] + self.gamma * self.tau * prev_A_sa * mask[t]
            
            # update previous
            prev_v_pseudo = v_pseudo[t]
            prev_A_sa = A_sa[t]
            prev_v = v[t]
                
            if t in idx or self.adv_est == 'ppo':
                v_target[t] = r[t] + self.gamma * prev_v_target * mask[t]
                prev_v_target = v_target[t]
            else:
                v_target[t] = prev_v_target

                
            if True:
                logging.debug('V:{:6.2f}-->{:6.2f} | Adv:{:6.2f} | r:{:3} {:4} {}'.format(
                       v[t].item(),
                       v_target[t].item(),
                       A_sa[t].item(),
                       '-' if r[t].item() == -0.025 else r[t].item(),
                       '' if t in idx else 'loop',
                       'END' if mask[t].item() == 0.0 else ''))
                
        # normalize A_sa
        exp_v = 1 - (v_target-v).std() / v_target.std()
        logging.debug('<<dialog policy ppo>> Adv mean {}, std {}'.format(A_sa.mean(), A_sa.std()))
        logging.debug('<<dialog policy ppo>> Value mean {}, std {}'.format(v_target.mean(), v_target.std()))
        logging.debug('<<dialog policy ppo>> Explained variance {}'.format(exp_v))
        A_sa = (A_sa - A_sa.mean()) / A_sa.std()
        return A_sa, v_target
    
    
    def est_adv_lcpo(self, r, v, mask, idx, max_r=1, verbose=True):
        """
        we save a trajectory in continuous space and it reaches the ending of current trajectory when mask=0.
        :param r: reward, Tensor, [b]
        :param v: estimated value, Tensor, [b]
        :param mask: indicates ending for 0 otherwise 1, Tensor, [b]
        :return: A(s, a), V-target(s), both Tensor
        """
        batchsz = v.size(0)

        # v_target is worked out by Bellman equation.
        v_target = torch.Tensor(batchsz).to(device=DEVICE)
        delta = torch.Tensor(batchsz).to(device=DEVICE)
        A_sa = torch.Tensor(batchsz).to(device=DEVICE)
        
        prev_v_target = 0
        prev_v = 0
        prev_A_sa = 0
        
        for t in reversed(range(batchsz)):
            if mask[t]==0:
                prev_v_target = v[t]
                prev_v = v[t]
                prev_A_sa = 0
            else:
                r[t] = -0.025
            
            bound    = (r[t] + self.gamma * v[t] * mask[t]) - v[t]
#             bound    = r[t] - v[t]
#             bound    = torch.tensor([-0.025])
            delta[t] = r[t] + self.gamma * prev_v * mask[t] - v[t]
            A_sa[t]  = delta[t] + self.gamma * self.tau * prev_A_sa * mask[t]
            
            if t in idx:
#             if True: # PPO
                # v_target[t] = A_sa[t] + v[t]
                v_target[t] = r[t] + self.gamma * prev_v_target * mask[t]
                A_sa[t] = max(bound, A_sa[t])
                
                # update previous
                prev_v_target = v_target[t]
                prev_v = v[t]
                prev_A_sa = A_sa[t]
            else:
                v_target[t] = prev_v_target
                A_sa[t] = min(bound, A_sa[t])
            
            if True:
                logging.debug('V:{:6.2f}-->{:6.2f} | Adv:{:6.2f} ({:5.2f}) | r:{:3} {:4} {}'.format(
                       v[t].item(),
                       v_target[t].item(),
                       A_sa[t].item(), 
                       bound.item(), 
                       '-' if r[t].item() == -0.025 else r[t].item(),
                       '' if t in idx else 'loop',
                       'END' if mask[t].item() == 0.0 else ''))
        # normalize A_sa
        A_sa = (A_sa - A_sa.mean()) / A_sa.std()

        return A_sa, v_target
    
    def loop_clipping(self, s, r, t, thresh=0.99, verbose=True):
        '''
        fixme if domain reward is used
        '''
        cos  = cosine_similarity(s)
        n_step = len(s)
        n_fail = 0
        loop   = [0]*n_step
        
        for i in range(n_step):
            if t[i]==0:
                if r[i] > 0:  # fixme if domain reward is used
                    continue
                else:
                    loop[i]=1
                    n_fail+=1
                    continue
                    
            for j in range(i+1,n_step):                
                if cos[i,j] > thresh:
                    loop[i:j]=[1]*(j-i)
                    break # save a bit of time
                if t[j]==0:
                    break
                
        idx = [ i for i,x in enumerate(loop) if x==0 ] # idx of turns which are not in loop
        
        if verbose:
            n_dialogue = t.numel() - t.nonzero().size(0)
            logging.debug('Success Rate: {:5.2}   {}/{} '.format(1-n_fail/n_dialogue, n_dialogue-n_fail, n_dialogue))
            logging.debug('Average Turn: {:5.2}'.format(n_step/n_dialogue))
            logging.debug('Looping Rate: {:5.2}   {}/{} '.format(1-len(idx)/n_step, n_step-len(idx), n_step))
            
        return idx
    
    
    def update(self, epoch, batchsz, s, a, r, mask):
        
        # get estimated V(s) and PI_old(s, a)
        # actually, PI_old(s, a) can be saved when interacting with env, so as to save the time of one forward elapsed
        # v: [b, 1] => [b]
        v = self.value(s).squeeze(-1).detach()
        log_pi_old_sa = self.policy.get_log_prob(s, a).detach()
        
        # estimate advantage and v_target according to GAE and Bellman Equation
        idx = self.loop_clipping(s,r,mask)  ### loop clipping HERE
        A_sa, v_target = self.est_adv(r, v, mask, idx)
        
        for i in range(self.update_round):

            # 1. shuffle current batch
            perm = torch.randperm(batchsz)
            # shuffle the variable for mutliple optimize
            v_target_shuf, A_sa_shuf, s_shuf, a_shuf, log_pi_old_sa_shuf = v_target[perm], A_sa[perm], s[perm], a[perm], \
                                                                           log_pi_old_sa[perm]

            # 2. get mini-batch for optimizing
            optim_chunk_num = int(np.ceil(batchsz / self.optim_batchsz))
            # chunk the optim_batch for total batch
            v_target_shuf, A_sa_shuf, s_shuf, a_shuf, log_pi_old_sa_shuf = torch.chunk(v_target_shuf, optim_chunk_num), \
                                                                           torch.chunk(A_sa_shuf, optim_chunk_num), \
                                                                           torch.chunk(s_shuf, optim_chunk_num), \
                                                                           torch.chunk(a_shuf, optim_chunk_num), \
                                                                           torch.chunk(log_pi_old_sa_shuf,
                                                                                       optim_chunk_num)
            # 3. iterate all mini-batch to optimize
            policy_loss, value_loss = 0., 0.
            for v_target_b, A_sa_b, s_b, a_b, log_pi_old_sa_b in zip(v_target_shuf, A_sa_shuf, s_shuf, a_shuf,
                                                                     log_pi_old_sa_shuf):
                # print('optim:', batchsz, v_target_b.size(), A_sa_b.size(), s_b.size(), a_b.size(), log_pi_old_sa_b.size())
                # 1. update value network
                self.value_optim.zero_grad()
                v_b = self.value(s_b).squeeze(-1)
#                 loss = F.binary_cross_entropy(v_b,v_target_b)
                loss = (v_b - v_target_b).pow(2).mean()
                value_loss += loss.item()
                
                # backprop
                loss.backward()
                # nn.utils.clip_grad_norm(self.value.parameters(), 4)
                self.value_optim.step()

                # 2. update policy network by clipping
                self.policy_optim.zero_grad()
                # [b, 1]
                log_pi_sa = self.policy.get_log_prob(s_b, a_b)
                # ratio = exp(log_Pi(a|s) - log_Pi_old(a|s)) = Pi(a|s) / Pi_old(a|s)
                # we use log_pi for stability of numerical operation
                # [b, 1] => [b]
                ratio = (log_pi_sa - log_pi_old_sa_b).exp().squeeze(-1)
                # because the joint action prob is the multiplication of the prob of each da
                # it may become extremely small
                # and the ratio may be inf in this case, which causes the gradient to be nan
                # clamp in case of the inf ratio, which causes the gradient to be nan
                ratio = torch.clamp(ratio, 0, 10)
                surrogate1 = ratio * A_sa_b
                surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * A_sa_b
                # this is element-wise comparing.
                # we add negative symbol to convert gradient ascent to gradient descent
                surrogate = - torch.min(surrogate1, surrogate2).mean()
                policy_loss += surrogate.item()

                # backprop
                surrogate.backward()
                # although the ratio is clamped, the grad may still contain nan due to 0 * inf
                # set the inf in the gradient to 0
                for p in self.policy.parameters():
                    p.grad[p.grad != p.grad] = 0.0
                # gradient clipping, for stability
                torch.nn.utils.clip_grad_norm(self.policy.parameters(), 10)
                # self.lock.acquire() # retain lock to update weights
                self.policy_optim.step()
                # self.lock.release() # release lock
            
            value_loss /= optim_chunk_num
            policy_loss /= optim_chunk_num
            logging.debug('<<dialog policy lcpo>> epoch {}, iteration {}, value, loss {}'.format(epoch, i, value_loss))
            logging.debug('<<dialog policy lcpo>> epoch {}, iteration {}, policy, loss {}'.format(epoch, i, policy_loss))

        if (epoch+1) % self.save_per_epoch == 0:
            self.save(self.save_dir, epoch)
    
    def save(self, directory, epoch):
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.value.state_dict(), directory + '/' + str(epoch) + '_lcpo.val.mdl')
        torch.save(self.policy.state_dict(), directory + '/' + str(epoch) + '_lcpo.pol.mdl')

        logging.info('<<dialog policy>> epoch {}: saved network to mdl'.format(epoch))
    
    def load(self, filename):
        value_mdl_candidates = [
            filename + '.val.mdl',
            filename + '_lcpo.val.mdl',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '.val.mdl'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_lcpo.val.mdl')
        ]
        for value_mdl in value_mdl_candidates:
            if os.path.exists(value_mdl):
                self.value.load_state_dict(torch.load(value_mdl, map_location=DEVICE))
                logging.info('<<dialog policy>> loaded checkpoint from file: {}'.format(value_mdl))
                break
        
        policy_mdl_candidates = [
            filename + '.pol.mdl',
            filename + '_lcpo.pol.mdl',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '.pol.mdl'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_lcpo.pol.mdl')
        ]
        for policy_mdl in policy_mdl_candidates:
            if os.path.exists(policy_mdl):
                self.policy.load_state_dict(torch.load(policy_mdl, map_location=DEVICE))
                logging.info('<<dialog policy>> loaded checkpoint from file: {}'.format(policy_mdl))
                break

    def load_from_pretrained(self, archive_file, model_file, filename):
        if not os.path.isfile(archive_file):
            if not model_file:
                raise Exception("No model for LCPO Policy is specified!")
            archive_file = cached_path(model_file)
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not os.path.exists(os.path.join(model_dir, 'best_lcpo.pol.mdl')):
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(model_dir)

        policy_mdl = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_lcpo.pol.mdl')
        if os.path.exists(policy_mdl):
            self.policy.load_state_dict(torch.load(policy_mdl, map_location=DEVICE))
            logging.info('<<dialog policy>> loaded checkpoint from file: {}'.format(policy_mdl))

        value_mdl = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_lcpo.val.mdl')
        if os.path.exists(value_mdl):
            self.value.load_state_dict(torch.load(value_mdl, map_location=DEVICE))
            logging.info('<<dialog policy>> loaded checkpoint from file: {}'.format(value_mdl))

#     @classmethod
#     def from_pretrained(cls,
#                         archive_file="",
#                         model_file="https://convlab.blob.core.windows.net/convlab-2/ppo_policy_multiwoz.zip",
#                         is_train=False,
#                         dataset='Multiwoz'):
#         with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'), 'r') as f:
#             cfg = json.load(f)
#         model = cls(is_train=is_train, dataset=dataset)
#         model.load_from_pretrained(archive_file, model_file, cfg['load'])
#         return model


# def loop_clipping_(self, s, r, t, thresh=0.99, verbose=True, success_reward=1):
#         n_step = len(s)
#         n_fail = 0
#         cos  = cosine_similarity(s)
#         idx  = []
#         ptr  = 0

#         for i in range(n_step):
#             if i<ptr:
#                 continue

#             for j in range(i+1,n_step):
#                 if cos[i,j] > thresh:
#                     ptr=j
#                     break  # mini-loop fixme max-loop

#                 if t[j]==0:
#                     break

#             if t[i]==0 and r[i]<success_reward:
#                 n_fail += 1
#             elif i>=ptr:
#                 idx.append(i)
        
#         if verbose:
# #             n_dialogue = t.count(0)
# #             n_loop = sum([len(l)for l in loop])
# #             assert n_step==n_loop+n_fail+len(idx)
#             n_dialogue = t.numel() - t.nonzero().size(0)
#             n_loop = n_step - n_fail - len(idx)
#             print ('Success Rate: {:5.2}   {}/{} '.format(1-n_fail/n_dialogue, n_dialogue-n_fail, n_dialogue))
#             print ('Average Turn: {:5.2}'.format(n_step/n_dialogue))
#             print ('Looping Rate: {:5.2}   {}+{}={}/{} '.format(1-len(idx)/n_step, n_fail, n_loop, n_fail+n_loop, n_step))
            
#         return idx