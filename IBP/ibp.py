import torch
import torch.nn as nn
import torch.nn.functional as F

def initial_bounds(x0, epsilon):
    '''
    x0 = input, b x c x h x w
    '''
    upper = x0+epsilon
    lower = x0-epsilon
    return upper, lower

def weighted_bound(layer, prev_upper, prev_lower):
    prev_mu = (prev_upper + prev_lower)/2
    prev_r = (prev_upper - prev_lower)/2
    mu = layer(prev_mu)
    if type(layer)==nn.Linear:
        r = F.linear(prev_r, torch.abs(layer.weight))
    elif type(layer)==nn.Conv2d:
        r = F.conv2d(prev_r, torch.abs(layer.weight), stride=layer.stride, padding=layer.padding)
    
    upper = mu + r
    lower = mu - r
    return upper, lower

def activation_bound(layer, prev_upper, prev_lower):
    upper = layer(prev_upper)
    lower = layer(prev_lower)
    return upper, lower

def network_bounds(model, x0, epsilon):
    '''
    get interval bound progation upper and lower bounds for the activation of a model
    
    model: a nn.Sequential module
    x0: input, b x input_shape
    epsilon: float, the linf distance bound is calculated over
    '''
    upper, lower = initial_bounds(x0, epsilon)
    for layer in model.modules():
        if type(layer) in (nn.Sequential,):
            pass
        elif type(layer) in (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.MaxPool2d, nn.Flatten):
            upper, lower = activation_bound(layer, upper, lower)
        elif type(layer) in (nn.Linear, nn.Conv2d):
            upper, lower = weighted_bound(layer, upper, lower)
        else:
            print('Unsupported layer:', type(layer))
    return upper, lower

def subsequent_bounds(model, upper, lower):
    '''
    get interval bound progation upper and lower bounds for the activation of a model,
    given bounds of the input
    
    model: a nn.Sequential module
    upper: upper bound on input layer, b x input_shape
    lower: lower bound on input layer, b x input_shape
    '''
    for layer in model.modules():
        if type(layer) in (nn.Sequential,):
            pass
        elif type(layer) in (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.MaxPool2d, nn.Flatten):
            upper, lower = activation_bound(layer, upper, lower)
        elif type(layer) in (nn.Linear, nn.Conv2d):
            upper, lower = weighted_bound(layer, upper, lower)
        else:
            print('Unsupported layer:', type(layer))
    return upper, lower


def worst_action_select(worst_q, upper_q, lower_q):
    mask = torch.zeros(upper_q.size()).cuda()
    for i in range(upper_q.size()[1]):
        upper = upper_q[:, i].view(upper_q.size()[0], 1)
        if_perturb = (upper > lower_q).all(1)
        mask[:, i] = if_perturb.byte()

    worst_q = worst_q.masked_fill_(mask==0, 1e9)
    worst_actions = worst_q.min(1)[-1].unsqueeze(1)
    worst_q = worst_q.gather(1, worst_actions).squeeze(1)

    return worst_q