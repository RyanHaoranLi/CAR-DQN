from __future__ import division
import time
import torch
import torch.nn.functional as F
import random
import logging
import numpy as np
from datetime import datetime
from copy import deepcopy

from utils import setup_logger
from plotter import plot
from ibp import network_bounds
from evaluate import eval_greedy_wc, eval_action_pert, attack_eval
from utils import EpsilonScheduler
from ibp import network_bounds, subsequent_bounds, worst_action_select

def _compute_car_loss(curr_model, target_model, data, epsilon, kappa, gamma, args, optimizer, epsilon_end=None):
    state, action, reward, next_state, done = data
    # adjust for the substracting mean in model.forward by making the model automatically substract the mean
    with torch.no_grad():
        curr_model.advantage[-1].weight[:] -= torch.mean(curr_model.advantage[-1].weight, dim=0, keepdim=True)
        curr_model.advantage[-1].bias[:] -= torch.mean(curr_model.advantage[-1].bias, dim=0, keepdim=True)

    value, advs = curr_model(state)
    q_values = value + advs

    next_value, next_advs = curr_model(next_state)
    next_q_values = next_value + next_advs

    target_next_value, target_next_advs = target_model(next_state)
    target_next_q_values = target_next_value + target_next_advs

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = target_next_q_values.gather(1, torch.argmax(next_q_values, 1, keepdim=True)).squeeze(1)
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    standard_loss = torch.min((q_value - expected_q_value.detach()).pow(2),
                              torch.abs(q_value - expected_q_value.detach()))
    standard_loss = standard_loss.mean()

    i_upper, i_lower = network_bounds(curr_model.cnn, state, epsilon)
    upper_adv, lower_adv = subsequent_bounds(curr_model.advantage, i_upper, i_lower)
    upper_value, lower_value = subsequent_bounds(curr_model.value, i_upper, i_lower)

    upper = upper_value + upper_adv
    lower = lower_value + lower_adv

    upper_td_error = upper.gather(1, action.unsqueeze(1)).squeeze(1) - expected_q_value.detach()
    lower_td_error = lower.gather(1, action.unsqueeze(1)).squeeze(1) - expected_q_value.detach()
    worst_case_td_error = torch.max(torch.abs(upper_td_error), torch.abs(lower_td_error))

    # Huber loss
    worst_case_td_error = torch.min(worst_case_td_error.pow(2), worst_case_td_error)

    soft_lambda = args.soft_lambda
    p = torch.exp(worst_case_td_error / soft_lambda).detach()
    p /= p.sum()
    worst_case_loss = torch.sum(p * worst_case_td_error)

    if args.trades:
        loss = kappa * (standard_loss) + (1 - kappa) * (worst_case_loss)
    else:
        loss = worst_case_loss

    return loss, standard_loss, worst_case_loss


def _compute_worst_loss(curr_model, worst_model, data, worst_eps, device, args):
    state, action, reward, next_state, done = data
    # adjust for the substracting mean in model.forward by making the model automatically substract the mean
    with torch.no_grad():
        curr_model.advantage[-1].weight[:] -= torch.mean(curr_model.advantage[-1].weight, dim=0, keepdim=True)
        curr_model.advantage[-1].bias[:] -= torch.mean(curr_model.advantage[-1].bias, dim=0, keepdim=True)
        worst_model.advantage[-1].weight[:] -= torch.mean(worst_model.advantage[-1].weight, dim=0, keepdim=True)
        worst_model.advantage[-1].bias[:] -= torch.mean(worst_model.advantage[-1].bias, dim=0, keepdim=True)

    worst_value, worst_advs = worst_model(state)
    worst_q_values = worst_value + worst_advs
    worst_q_value = worst_q_values.gather(1, action.unsqueeze(1)).squeeze(1)

    next_worst_value, next_worst_advs = worst_model(next_state)
    next_worst_q_values = next_worst_value + next_worst_advs

    i_upper, i_lower = network_bounds(curr_model.cnn, state, 1e-3)
    upper_adv, lower_adv = subsequent_bounds(curr_model.advantage, i_upper, i_lower)
    upper_value, lower_value = subsequent_bounds(curr_model.value, i_upper, i_lower)
    curr_q_values_ub = upper_value + upper_adv
    curr_q_values_lb = lower_value + lower_adv

    worst_next_q_value = worst_action_select(next_worst_q_values, curr_q_values_ub, curr_q_values_lb)

    exp_worst_q_value = reward + args.gamma * worst_next_q_value * (1 - done)
    # Huber loss
    worst_loss = torch.min((worst_q_value - exp_worst_q_value.detach()).pow(2),
                           torch.abs(worst_q_value - exp_worst_q_value.detach()))
    worst_loss = worst_loss.mean()

    return worst_loss


def _compute_wocar_loss(curr_model, target_model, worst_model, data, epsilon, kappa, gamma, device, args):
    state, action, reward, next_state, done = data
    # adjust for the substracting mean in model.forward by making the model automatically substract the mean
    with torch.no_grad():
        curr_model.advantage[-1].weight[:] -= torch.mean(curr_model.advantage[-1].weight, dim=0, keepdim=True)
        curr_model.advantage[-1].bias[:] -= torch.mean(curr_model.advantage[-1].bias, dim=0, keepdim=True)
        worst_model.advantage[-1].weight[:] -= torch.mean(worst_model.advantage[-1].weight, dim=0, keepdim=True)
        worst_model.advantage[-1].bias[:] -= torch.mean(worst_model.advantage[-1].bias, dim=0, keepdim=True)

    value, advs = curr_model(state)
    q_values = value + advs
    next_value, next_advs = curr_model(next_state)
    next_q_values = next_value + next_advs
    target_value, target_advs = target_model(next_state)
    next_q_state_values = target_value + target_advs
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

    worst_value, worst_advs = worst_model(next_state)
    worst_q_values = worst_value + worst_advs

    weighted_next_q_values = (1 - args.worst_ratio) * next_q_values + args.worst_ratio * worst_q_values
    _, next_max_action = torch.max(weighted_next_q_values, dim=1)

    next_q_value = next_q_state_values.gather(1, next_max_action.unsqueeze(1)).squeeze(1)

    expected_q_value = reward + gamma * next_q_value * (1 - done)

    standard_loss = torch.min((q_value - expected_q_value.detach()).pow(2),
                              torch.abs(q_value - expected_q_value.detach()))

    i_upper, i_lower = network_bounds(curr_model.cnn, state, epsilon)
    upper_adv, lower_adv = subsequent_bounds(curr_model.advantage, i_upper, i_lower)
    upper_value, lower_value = subsequent_bounds(curr_model.value, i_upper, i_lower)
    upper = upper_value + upper_adv
    lower = lower_value + lower_adv
    onehot_labels = torch.zeros(upper.shape).to(device)
    onehot_labels[range(state.shape[0]), action] = 1

    if args.worse_bound:
        upper_diff = upper - q_values * (1 - onehot_labels) - expected_q_value.detach().unsqueeze(1) * onehot_labels
        lower_diff = lower - q_values * (1 - onehot_labels) - expected_q_value.detach().unsqueeze(1) * onehot_labels
        wc_diff = torch.max(torch.abs(upper_diff), torch.abs(lower_diff))
    else:
        worst_case = onehot_labels * lower + (1 - onehot_labels) * upper
        wc_diff = torch.abs(
            worst_case - q_values * (1 - onehot_labels) - expected_q_value.detach().unsqueeze(1) * onehot_labels)

    # sum over output layer, mean only in batch dimension
    worst_case_loss = torch.sum(torch.min(wc_diff.pow(2), wc_diff), dim=1).mean()

    standard_loss = standard_loss.mean()

    loss = (kappa * (standard_loss) + (1 - kappa) * (worst_case_loss))

    return loss, standard_loss, worst_case_loss


def _compute_robust_loss(curr_model, target_model, data, epsilon, kappa, gamma, device, args, epsilon_end=None):
    state, action, reward, next_state, done = data
    # adjust for the substracting mean in model.forward by making the model automatically substract the mean
    with torch.no_grad():
        curr_model.advantage[-1].weight[:] -= torch.mean(curr_model.advantage[-1].weight, dim=0, keepdim=True)
        curr_model.advantage[-1].bias[:] -= torch.mean(curr_model.advantage[-1].bias, dim=0, keepdim=True)

    value, advs = curr_model(state)
    q_values = value + advs

    next_value, next_advs = curr_model(next_state)
    next_q_values = next_value + next_advs

    target_next_value, target_next_advs = target_model(next_state)
    target_next_q_values = target_next_value + target_next_advs

    q_value      = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = target_next_q_values.gather(1, torch.argmax(next_q_values, 1, keepdim=True)).squeeze(1)
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    standard_loss = torch.min((q_value - expected_q_value.detach()).pow(2), torch.abs(q_value - expected_q_value.detach()))
    
    i_upper, i_lower = network_bounds(curr_model.cnn, state, epsilon)
    upper, lower = subsequent_bounds(curr_model.advantage, i_upper, i_lower)

    upper += value.detach()
    lower += value.detach()
        
    onehot_labels = torch.zeros(upper.shape).to(device)
    onehot_labels[range(state.shape[0]), action] = 1

    if args.loss_fn == "approach_1":
        upper_diff = (upper - expected_q_value.detach().unsqueeze(1))*onehot_labels + (upper - q_values)*(1-onehot_labels) 
        lower_diff = (lower - expected_q_value.detach().unsqueeze(1))*onehot_labels + (lower - q_values)*(1-onehot_labels) 
        wc_diff = torch.max(torch.abs(upper_diff), torch.abs(lower_diff))
        worst_case_loss = torch.sum(torch.min(wc_diff.pow(2), wc_diff), dim=1).mean()
    
    elif args.loss_fn == "approach_2":
        # calculate how much worse each action is than the action taken
        q_diff = torch.max(torch.zeros([1]).to(device), 
                (q_values.gather(1, action.unsqueeze(1)).detach()-q_values.detach()))
        overlap = torch.max(upper - lower.gather(1, action.unsqueeze(1)) + q_diff.detach()/2, torch.zeros([1]).to(device))
        worst_case_loss = torch.sum(q_diff*overlap, dim=1).mean()+1e-4*torch.ones([1]).to(device)

    standard_loss = standard_loss.mean()
    loss = kappa*(standard_loss)+(1-kappa)*(worst_case_loss)
    
    return loss, standard_loss, worst_case_loss

def _compute_loss(curr_model, target_model, data, gamma, device):
    state, action, reward, next_state, done = data
    
    value, advs = curr_model(state)
    q_values = value + advs

    next_value, next_advs = curr_model(next_state)
    next_q_values = next_value + next_advs

    target_next_value, target_next_advs = target_model(next_state)
    target_next_q_values = target_next_value + target_next_advs

    q_value       = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = target_next_q_values.gather(1, torch.argmax(next_q_values, 1, keepdim=True)).squeeze(1)
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    #Huber loss
    standard_loss = torch.min((q_value - expected_q_value.detach()).pow(2), torch.abs(q_value - expected_q_value.detach()))
    standard_loss = standard_loss.mean()
    
    return standard_loss, standard_loss, standard_loss


def car_train(current_model, target_model, env, args):
    start_time = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    log = {}
    setup_logger('{}_log'.format(args.env), r'{}{}_{}_log'.format(
        args.log_dir, args.env, start_time))
    log['{}_log'.format(args.env)] = logging.getLogger('{}_log'.format(
        args.env))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    # linearly decrease epsilon from 1 to epsilon end over epsilon decay steps
    epsilon_start = 1.0

    def action_epsilon_by_frame(frame_idx):
        if frame_idx <= args.exp_epsilon_decay or not args.decay_zero:
            return (args.exp_epsilon_end + max(0, 1 - frame_idx / args.exp_epsilon_decay) * (
                        epsilon_start - args.exp_epsilon_end))
        else:
            return max(0, (args.total_frames - frame_idx) / (args.total_frames - args.exp_epsilon_decay)) * (
                args.exp_epsilon_end)

    if args.gpu_id >= 0:
        device = torch.device('cuda:{}'.format(args.gpu_id))
    else:
        device = torch.device('cpu')

    replay_buffer = ReplayBuffer(args.buffer_size, device)
    start = time.time()

    losses = []
    standard_losses = []
    worst_case_losses = []
    all_rewards = []
    worst_case_rewards = []
    episode_reward = 0

    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    best_avg = float("-inf")

    if args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(current_model.parameters(), lr=args.lr, momentum=0.95, alpha=0.95, eps=1e-2)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(current_model.parameters(), lr=args.lr, amsgrad=args.amsgrad, eps=args.adam_eps)

    if args.smoothed:
        attack_eps_scheduler = EpsilonScheduler("smoothed", args.replay_initial, args.attack_epsilon_schedule, 0,
                                                args.attack_epsilon_end, args.attack_epsilon_schedule)

    start_step = 1
    if args.resume_training:
        saved_state = torch.load(
            args.resume_training,
            map_location=lambda storage, loc: storage)
        current_model.load_state_dict(saved_state['model_state_dict'])
        target_model.load_state_dict(current_model.state_dict())

        optimizer.load_state_dict(saved_state['optimizer_state_dict'])
        # move start state back to allow for regenerating a replay buffer without skipping training
        start_step = saved_state['step'] - args.replay_initial

    all_natural_return = []
    all_robust_return = []

    # main loop
    episode_count = 0
    for frame_idx in range(start_step, args.total_frames + 1):
        action_epsilon = action_epsilon_by_frame(frame_idx)
        action = current_model.act(state, action_epsilon)
        next_state, reward, done, info = env.step(action)
        episode_reward += reward

        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        action = torch.LongTensor([action]).to(device)
        # scale rewards between -1 and 1
        reward = torch.clamp(torch.FloatTensor([reward]).to(device), min=-1, max=1)
        done = torch.FloatTensor([done]).to(device)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state

        # test the model every 100000 frames
        if frame_idx % 100000 == 0:
            if args.smoothed:
                attack_epsilon = attack_eps_scheduler.get_eps(0, frame_idx)
            else:
                init_coeff = (frame_idx - args.replay_initial + 1) / min(args.attack_epsilon_schedule,
                                                                         args.total_frames)
                lin_coeff = max(min(1, init_coeff), 0)
                attack_epsilon = lin_coeff * args.attack_epsilon_end

            test_rewards = []
            for i in range(10):
                test_rewards.append(eval_action_pert(current_model, deepcopy(env), args, epsilon=0))
            avg_reward = np.mean(test_rewards)
            log['{}_log'.format(args.env)].info("Steps: {}, Avg reward: {:.2f}, Time taken: {:.2f}s".format(
                frame_idx, avg_reward, time.time() - start))
            # all_natural_return.append(avg_reward)

            # save the best model
            if np.mean(test_rewards) >= best_avg:
                state_to_save = {'model_state_dict': current_model.state_dict(),
                                 'optimizer_state_dict': optimizer.state_dict(),
                                 'step': frame_idx}
                best_avg = np.mean(test_rewards)
                if args.resume_training:
                    torch.save(state_to_save, args.resume_training[:-3] + "_best.pt")
                else:
                    torch.save(state_to_save, '{}{}_{}_{}_best.pt'.format(
                        args.save_model_dir, args.env, args.loss_fn, start_time))

            # # robust test
            # robust_test_rewards = []
            # for i in range(20):
            #     robust_test_rewards.append(attack_eval(current_model, deepcopy(env), args, args.attack_epsilon_end, 'PGD'))
            # avg_robust_reward = np.mean(robust_test_rewards)
            # log['{}_log'.format(args.env)].info("Steps: {}, Avg robust reward: {:.2f}, Time taken: {:.2f}s".format(
            #     frame_idx, avg_robust_reward, time.time() - start))
            # all_robust_return.append(avg_robust_reward)

            # np.save('{}{}_all_natural_return.npy'.format(args.save_model_dir, start_time), all_natural_return)
            # np.save('{}{}_all_robust_return.npy'.format(args.save_model_dir, start_time), all_robust_return)

        elif frame_idx % 10000 == 0:
            test_reward = eval_action_pert(current_model, deepcopy(env), args, epsilon=0)
            log['{}_log'.format(args.env)].info("Steps: {}, Test reward: {}, Time taken: {:.2f}s".format(frame_idx, test_reward, time.time() - start))

        # save the last model
        if done:
            state = env.reset()
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            all_rewards.append(episode_reward)
            # log['{}_log'.format(args.env)].info("Episode: {}, Episode reward: {}, Steps: {}, Time taken: {:.2f}s".format(episode_count, episode_reward, frame_idx, time.time() - start))
            episode_reward = 0
            plot(frame_idx, all_rewards, losses, standard_losses, worst_case_losses, args, start_time)

            state_to_save = {'model_state_dict': current_model.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),
                             'step': frame_idx}

            # if frame_idx > (args.replay_initial + args.attack_epsilon_schedule):
            #     if args.resume_training:
            #         torch.save(state_to_save, args.resume_training[:-3] + "_last.pt")
            #     else:
            #         torch.save(state_to_save, '{}{}_{}_{}_last_{}.pt'.format(
            #             args.save_model_dir, args.env, args.loss_fn, start_time, frame_idx))

            if args.resume_training:
                torch.save(state_to_save, args.resume_training[:-3] + "_last.pt")
            else:
                torch.save(state_to_save, '{}{}_{}_{}_last.pt'.format(
                    args.save_model_dir, args.env, args.loss_fn, start_time))
            episode_count += 1

        # update the model
        if len(replay_buffer) > args.replay_initial and frame_idx % (args.batch_size / args.updates_per_frame) == 0:

            init_coeff = (frame_idx - args.replay_initial + 1) / min(args.attack_epsilon_schedule, args.total_frames)
            # clip between 0 and 1
            lin_coeff = max(min(1, init_coeff), 0)

            if args.smoothed:
                attack_epsilon = attack_eps_scheduler.get_eps(0, frame_idx)
            else:
                attack_epsilon = lin_coeff * args.attack_epsilon_end

            if args.constant_kappa:
                kappa = args.kappa_end
            else:
                kappa = (1 - lin_coeff) * 1 + lin_coeff * args.kappa_end

            data = replay_buffer.sample(args.batch_size)
            loss, standard_loss, worst_case_loss = _compute_car_loss(current_model, target_model, data, attack_epsilon, kappa, args.gamma, args, optimizer, args.attack_epsilon_end)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.data.item())
            standard_losses.append(standard_loss.data.item())
            worst_case_losses.append(worst_case_loss.data.item())

        if frame_idx % (1000 * (args.batch_size / args.updates_per_frame)) == 0:
            target_model.load_state_dict(current_model.state_dict())

    # save the final model
    state_to_save = {'model_state_dict': current_model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'step': frame_idx}
    if args.resume_training:
        torch.save(state_to_save, args.resume_training[:-3] + "_final.pt")
    else:
        torch.save(state_to_save, '{}{}_{}_{}_final.pt'.format(
            args.save_model_dir, args.env, args.loss_fn, start_time))

    log['{}_log'.format(args.env)].info("Done in {:.3f}s".format(time.time() - start))
    np.save('{}losses.npy'.format(args.save_model_dir), losses)
    np.save('{}standard_losses.npy'.format(args.save_model_dir), standard_losses)
    np.save('{}worst_case_losses.npy'.format(args.save_model_dir), worst_case_losses)
    np.save('{}all_rewards.npy'.format(args.save_model_dir), all_rewards)


def wocar_train(current_model, target_model, worst_model, env, args):
    start_time = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    log = {}
    setup_logger('{}_log'.format(args.env), r'{}{}_{}_log'.format(
        args.log_dir, args.env, start_time))
    log['{}_log'.format(args.env)] = logging.getLogger('{}_log'.format(
        args.env))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    training_count = 0
    training_worst = 0
    # linearly decrease epsilon from 1 to epsilon end over epsilon decay steps
    epsilon_start = 1.0

    def action_epsilon_by_frame(frame_idx):
        if frame_idx <= args.exp_epsilon_decay or not args.decay_zero:
            return (args.exp_epsilon_end + max(0, 1 - frame_idx / args.exp_epsilon_decay) * (
                        epsilon_start - args.exp_epsilon_end))
        else:
            return max(0, (args.total_frames - frame_idx) / (args.total_frames - args.exp_epsilon_decay)) * (
                args.exp_epsilon_end)

    if args.gpu_id >= 0:
        device = torch.device('cuda:{}'.format(args.gpu_id))
    else:
        device = torch.device('cpu')

    replay_buffer = ReplayBuffer(args.buffer_size, device)
    start = time.time()

    losses = []
    standard_losses = []
    worst_case_losses = []
    worst_losses = []
    all_rewards = []
    worst_case_rewards = []
    # initialize as a large negative number to save first
    max_score = -10000
    episode_reward = 0

    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    best_avg = float("-inf")

    if args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(current_model.parameters(), lr=args.lr, momentum=0.95, alpha=0.95, eps=1e-2)
        worst_optimizer = torch.optim.RMSprop(worst_model.parameters(), lr=args.lr, momentum=0.95, alpha=0.95, eps=1e-2)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(current_model.parameters(), lr=args.lr, amsgrad=args.amsgrad, eps=args.adam_eps)
        worst_optimizer = torch.optim.Adam(worst_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)

    if args.smoothed:
        attack_eps_scheduler = EpsilonScheduler("smoothed", args.replay_initial, args.attack_epsilon_schedule, 0,
                                                args.attack_epsilon_end, args.attack_epsilon_schedule)

    start_step = 1
    if args.resume_training:
        saved_state = torch.load(
            args.resume_training,
            map_location=lambda storage, loc: storage)
        current_model.load_state_dict(saved_state['model_state_dict'])
        target_model.load_state_dict(current_model.state_dict())

        optimizer.load_state_dict(saved_state['optimizer_state_dict'])
        # move start state back to allow for regenerating a replay buffer without skipping training
        start_step = saved_state['step'] - args.replay_initial

    all_natural_return = []
    all_robust_return = []

    # main loop
    for frame_idx in range(start_step, args.total_frames + 1):
        action_epsilon = action_epsilon_by_frame(frame_idx)
        action = current_model.act(state, action_epsilon)
        next_state, reward, done, info = env.step(action)
        episode_reward += reward

        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        action = torch.LongTensor([action]).to(device)
        # scale rewards between -1 and 1
        reward = torch.clamp(torch.FloatTensor([reward]).to(device), min=-1, max=1)
        done = torch.FloatTensor([done]).to(device)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state

        # test the model
        if frame_idx % 100000 == 0:
            if args.smoothed:
                attack_epsilon = attack_eps_scheduler.get_eps(0, frame_idx)
            else:
                init_coeff = (frame_idx - args.replay_initial + 1) / min(args.attack_epsilon_schedule,
                                                                         args.total_frames)
                lin_coeff = max(min(1, init_coeff), 0)
                attack_epsilon = lin_coeff * args.attack_epsilon_end

            test_rewards = []
            for i in range(10):
                test_rewards.append(eval_action_pert(current_model, deepcopy(env), args, epsilon=0))
            avg_reward = np.mean(test_rewards)
            log['{}_log'.format(args.env)].info("Steps: {}, Avg reward: {:.2f}, Time taken: {:.2f}s".format(
                frame_idx, avg_reward, time.time() - start))
            # all_natural_return.append(avg_reward)

            # # robust test
            # robust_test_rewards = []
            # for i in range(20):
            #     robust_test_rewards.append(
            #         attack_eval(current_model, deepcopy(env), args, args.attack_epsilon_end, 'PGD'))
            # avg_robust_reward = np.mean(robust_test_rewards)
            # log['{}_log'.format(args.env)].info("Steps: {}, Avg robust reward: {:.2f}, Time taken: {:.2f}s".format(
            #     frame_idx, avg_robust_reward, time.time() - start))
            # all_robust_return.append(avg_robust_reward)

            # np.save('{}{}_all_natural_return.npy'.format(args.save_model_dir, start_time), all_natural_return)
            # np.save('{}{}_all_robust_return.npy'.format(args.save_model_dir, start_time), all_robust_return)

            # save the best model
            if np.mean(test_rewards) >= best_avg:
                state_to_save = {'model_state_dict': current_model.state_dict(),
                                 'optimizer_state_dict': optimizer.state_dict(),
                                 'step': frame_idx}
                best_avg = np.mean(test_rewards)
                if args.resume_training:
                    torch.save(state_to_save, args.resume_training[:-3] + "_best.pt")
                else:
                    torch.save(state_to_save, '{}{}_{}_{}_best.pt'.format(
                        args.save_model_dir, args.env, args.loss_fn, start_time))

        elif frame_idx % 10000 == 0:
            test_reward = eval_action_pert(current_model, deepcopy(env), args, epsilon=0)
            log['{}_log'.format(args.env)].info("Steps: {}, Test reward: {}, Time taken: {:.2f}s".format(frame_idx,
                                                                                                         test_reward,
                                                                                                         time.time() - start))

        # save the last model
        if done:
            state = env.reset()
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            all_rewards.append(episode_reward)
            episode_reward = 0
            plot(frame_idx, all_rewards, losses, standard_losses, worst_case_losses, args, start_time)

            state_to_save = {'model_state_dict': current_model.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),
                             'step': frame_idx}

            if args.resume_training:
                torch.save(state_to_save, args.resume_training[:-3] + "_last.pt")
            else:
                torch.save(state_to_save, '{}{}_{}_{}_last.pt'.format(
                    args.save_model_dir, args.env, args.loss_fn, start_time))

        # update the model
        if len(replay_buffer) > args.replay_initial and frame_idx % (args.batch_size / args.updates_per_frame) == 0:

            init_coeff = (frame_idx - args.replay_initial + 1) / min(args.attack_epsilon_schedule, args.total_frames)
            # clip between 0 and 1
            lin_coeff = max(min(1, init_coeff), 0)

            if args.smoothed:
                attack_epsilon = attack_eps_scheduler.get_eps(0, frame_idx)
            else:
                attack_epsilon = lin_coeff * args.attack_epsilon_end

            if args.constant_kappa:
                kappa = args.kappa_end
            else:
                kappa = (1 - lin_coeff) * 1 + lin_coeff * args.kappa_end

            data = replay_buffer.sample(args.batch_size)
            if training_count % args.worst_rate == 0:
                training_worst += 1
                if training_worst % 100 == 0:
                    print("training_count:", training_count)
                    print("training_worst:", training_worst)
                worst_loss = _compute_worst_loss(current_model, worst_model, data, attack_epsilon, device, args)
                worst_optimizer.zero_grad()
                worst_loss.backward()
                worst_optimizer.step()
                worst_losses.append(worst_loss.data.item())

            training_count += 1
            loss, standard_loss, worst_case_loss = _compute_wocar_loss(current_model, target_model, worst_model, data,
                                                                       attack_epsilon,
                                                                       kappa, args.gamma, device, args)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.data.item())
            standard_losses.append(standard_loss.data.item())
            worst_case_losses.append(worst_case_loss.data.item())

        if frame_idx % (1000 * (args.batch_size / args.updates_per_frame)) == 0:
            target_model.load_state_dict(current_model.state_dict())

    # save final model
    state_to_save = {'model_state_dict': current_model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'step': frame_idx}
    if args.resume_training:
        torch.save(state_to_save, args.resume_training[:-3] + "_last.pt")
    else:
        torch.save(state_to_save, '{}{}_{}_{}_last.pt'.format(
            args.save_model_dir, args.env, args.loss_fn, start_time))

    log['{}_log'.format(args.env)].info("Done in {:.3f}s".format(time.time() - start))


def train(current_model, target_model, env, args):
    start_time = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    log = {}
    setup_logger('{}_log'.format(args.env), r'{}{}_{}_log'.format(
        args.log_dir, args.env, start_time))
    log['{}_log'.format(args.env)] = logging.getLogger('{}_log'.format(
        args.env))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))
    
    
    # linearly decrease epsilon from 1 to epsilon end over epsilon decay steps
    epsilon_start = 1.0
    def action_epsilon_by_frame(frame_idx):
        if frame_idx <= args.exp_epsilon_decay or not args.decay_zero:
            return (args.exp_epsilon_end + max(0, 1-frame_idx/args.exp_epsilon_decay)*(epsilon_start-args.exp_epsilon_end))
        else:
            return max(0, (args.total_frames-frame_idx) / (args.total_frames-args.exp_epsilon_decay))*(args.exp_epsilon_end)
    
    if args.gpu_id >= 0:
        device = torch.device('cuda:{}'.format(args.gpu_id))
    else:
        device = torch.device('cpu')
    
    replay_buffer = ReplayBuffer(args.buffer_size, device)
    start = time.time()
    
    losses = []
    standard_losses = []
    worst_case_losses = []
    all_rewards = []
    worst_case_rewards = []
    episode_reward = 0
    
    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    best_avg = float("-inf")
    
    if args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(current_model.parameters(), lr=args.lr, momentum=0.95, alpha=0.95, eps=1e-2)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(current_model.parameters(), lr=args.lr, amsgrad=args.amsgrad, eps=args.adam_eps)

    if args.smoothed:
        attack_eps_scheduler = EpsilonScheduler("smoothed", args.replay_initial, args.attack_epsilon_schedule, 0, 
                                                args.attack_epsilon_end, args.attack_epsilon_schedule)
    
    start_step = 1
    if args.resume_training:
        saved_state = torch.load(
            args.resume_training,
            map_location=lambda storage, loc: storage)
        current_model.load_state_dict(saved_state['model_state_dict'])
        target_model.load_state_dict(current_model.state_dict())
        
        optimizer.load_state_dict(saved_state['optimizer_state_dict'])
        # move start state back to allow for regenerating a replay buffer without skipping training
        start_step = saved_state['step'] - args.replay_initial

    all_natural_return = []
    all_robust_return = []

    # main loop
    for frame_idx in range(start_step, args.total_frames + 1):
        action_epsilon = action_epsilon_by_frame(frame_idx)
        action = current_model.act(state, action_epsilon)
        next_state, reward, done, info = env.step(action)
        episode_reward += reward

        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        action = torch.LongTensor([action]).to(device)
        # scale rewards between -1 and 1
        reward = torch.clamp(torch.FloatTensor([reward]).to(device), min=-1, max=1)
        done =  torch.FloatTensor([done]).to(device)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state

        # test the model
        if frame_idx%100000==0:
            if args.smoothed:
                attack_epsilon = attack_eps_scheduler.get_eps(0, frame_idx)
            else:
                init_coeff = (frame_idx - args.replay_initial +1)/min(args.attack_epsilon_schedule, args.total_frames)
                lin_coeff = max(min(1, init_coeff), 0)
                attack_epsilon = lin_coeff*args.attack_epsilon_end

            test_rewards = []
            for i in range(10):
                test_rewards.append(eval_action_pert(current_model, deepcopy(env), args, epsilon=0))
            avg_reward = np.mean(test_rewards)
            log['{}_log'.format(args.env)].info("Steps: {}, Avg reward: {:.2f}, Time taken: {:.2f}s".format(
                frame_idx, avg_reward, time.time() - start))
            # all_natural_return.append(avg_reward)

            # # robust test
            # robust_test_rewards = []
            # for i in range(20):
            #     robust_test_rewards.append(
            #         attack_eval(current_model, deepcopy(env), args, args.attack_epsilon_end, 'PGD'))
            # avg_robust_reward = np.mean(robust_test_rewards)
            # log['{}_log'.format(args.env)].info("Steps: {}, Avg robust reward: {:.2f}, Time taken: {:.2f}s".format(
            #     frame_idx, avg_robust_reward, time.time() - start))
            # all_robust_return.append(avg_robust_reward)

            # np.save('{}{}_all_natural_return.npy'.format(args.save_model_dir, start_time), all_natural_return)
            # np.save('{}{}_all_robust_return.npy'.format(args.save_model_dir, start_time), all_robust_return)

            if args.robust:
                gwc = eval_greedy_wc(current_model, deepcopy(env), args, attack_epsilon)
                log['{}_log'.format(args.env)].info("GWC: {:.1f} at epsilon: {:.6f}, Time taken: {:.2f}s".format(gwc, 
                            attack_epsilon, time.time()-start))
                # if gwc not good, get gwc for smaller epsilon. Add 21 to make all rewards be on positive scale
                while gwc+21 < 0.9*(avg_reward+21):
                    attack_epsilon = attack_epsilon/2.0
                    gwc = eval_greedy_wc(current_model, deepcopy(env), args, attack_epsilon)
                    log['{}_log'.format(args.env)].info("GWC: {:.1f} at epsilon: {:.6f}, Time taken: {:.2f}s".format(gwc, 
                            attack_epsilon, time.time()-start))

            # save the best model
            if np.mean(test_rewards) >= best_avg:
                state_to_save = {'model_state_dict':current_model.state_dict(),
                            'optimizer_state_dict':optimizer.state_dict(),
                            'step':frame_idx}
                best_avg = np.mean(test_rewards)
                if args.resume_training:
                    torch.save(state_to_save, args.resume_training[:-3]+"_best.pt")
                else:
                    torch.save(state_to_save, '{}{}_{}_{}_best.pt'.format(
                        args.save_model_dir, args.env, args.loss_fn, start_time))
        
        elif frame_idx%10000==0:
            test_reward = eval_action_pert(current_model, deepcopy(env), args, epsilon=0)
            log['{}_log'.format(args.env)].info("Steps: {}, Test reward: {}, Time taken: {:.2f}s".format(frame_idx,
                                                                                   test_reward, time.time()-start))

        # save the last model
        if done:
            state = env.reset()
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            all_rewards.append(episode_reward)
            episode_reward = 0
            plot(frame_idx, all_rewards, losses, standard_losses, worst_case_losses, args, start_time)
            
            state_to_save = {'model_state_dict':current_model.state_dict(),
                            'optimizer_state_dict':optimizer.state_dict(),
                            'step':frame_idx}
            
            if args.resume_training:
                torch.save(state_to_save, args.resume_training[:-3]+"_last.pt")
            else:
                torch.save(state_to_save, '{}{}_{}_{}_last.pt'.format(
                    args.save_model_dir, args.env, args.loss_fn, start_time))
            
        # update the model
        if len(replay_buffer) > args.replay_initial and frame_idx%(args.batch_size/args.updates_per_frame)==0:
            
            init_coeff = (frame_idx - args.replay_initial +1)/min(args.attack_epsilon_schedule, args.total_frames)
            # clip between 0 and 1
            lin_coeff = max(min(1, init_coeff), 0)
            
            if args.smoothed:
                attack_epsilon = attack_eps_scheduler.get_eps(0, frame_idx)
            else:
                attack_epsilon = lin_coeff*args.attack_epsilon_end
            
            if args.constant_kappa:
                kappa = args.kappa_end
            else:
                kappa = (1-lin_coeff)*1 + lin_coeff*args.kappa_end

            data = replay_buffer.sample(args.batch_size)
            if args.robust:
                loss, standard_loss, worst_case_loss = _compute_robust_loss(current_model, target_model, data, attack_epsilon,
                                                                     kappa, args.gamma, device, args, args.attack_epsilon_end)
            
            else:
                loss, standard_loss, worst_case_loss = _compute_loss(current_model, target_model, data, args.gamma, device)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.data.item())
            standard_losses.append(standard_loss.data.item())
            worst_case_losses.append(worst_case_loss.data.item())

        if frame_idx % (1000*(args.batch_size/args.updates_per_frame)) == 0:
            target_model.load_state_dict(current_model.state_dict())
            
    # save final model
    state_to_save = {'model_state_dict':current_model.state_dict(),
                     'optimizer_state_dict':optimizer.state_dict(),
                     'step':frame_idx}
    if args.resume_training:
        torch.save(state_to_save, args.resume_training[:-3]+"_last.pt")
    else:
        torch.save(state_to_save, '{}{}_{}_{}_last.pt'.format(
            args.save_model_dir, args.env, args.loss_fn, start_time))
                
    log['{}_log'.format(args.env)].info("Done in {:.3f}s".format(time.time()-start))
    
    
class ReplayBuffer(object):
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.buffer = []
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (torch.cat(state, dim=0), torch.cat(action, dim=0), torch.cat(reward, dim=0), 
                torch.cat(next_state, dim=0), torch.cat(done, dim=0))
        
    def __len__(self):
        return len(self.buffer)