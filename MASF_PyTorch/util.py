import torch

def kd(prob1, prob2):
    prob1 = torch.clamp(prob1, min=1e-8, max=1.0)
    prob2 = torch.clamp(prob2, min=1e-8, max=1.0)
    return (torch.mean(prob1 * torch.log(prob1 / prob2)) + torch.mean(prob2 * torch.log(prob2 / prob1))) / 2.0

def distance_function(m_0, m_1):
    return torch.norm(torch.sub(m_0, m_1))**2


def triple_loss(a, p, n, margin=0.2) : 
    d = torch.nn.PairwiseDistance(p=2)
    distance = distance_function(a, p) - distance_function(a, n) + margin 
    loss = torch.mean(torch.max(distance, torch.zeros(1).cuda()))
    return loss
    