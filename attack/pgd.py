import torch
import numpy as np


class PGD_Attack_L2():
    def __init__(self, model, attack_criterion, 
                num_channels, input_shape, mean, std,
                step_size=0.1, epsilon=1., iterations=100,
                return_normalized=True, img_min=0, img_max=1, cuda=True):
        """
        PGD L2 attack
        
        Resources for reference:
            https://github.com/MadryLab/robustness/blob/master/robustness/attacker.py
            https://github.com/MadryLab/robustness/blob/master/robustness/attack_steps.py
        """
        self.model = model
        # Model should be in evaluation mode
        self.model.eval()

        self.attack_criterion = attack_criterion
        
        self.mean_tensor = torch.ones(1, num_channels, *input_shape)
        for idx in range(num_channels):
            self.mean_tensor[:,idx] *= mean[idx]
        if cuda:
            self.mean_tensor = self.mean_tensor.cuda()

        self.std_tensor = torch.ones(1, num_channels, *input_shape)
        for idx in range(num_channels):
            self.std_tensor[:,idx] *= std[idx]
        if cuda:
            self.std_tensor = self.std_tensor.cuda()
        
        self.step_size = step_size 
        self.epsilon = epsilon
        self.iterations = iterations
        self.return_normalized = return_normalized
        self.img_min = img_min 
        self.img_max = img_max
        
    def run(self, orig_img, lbl):
        # Unnormalize
        img = orig_img * self.std_tensor + self.mean_tensor
        
        x = img.clone().detach().requires_grad_(True)
        
        for i in range(self.iterations):
            x = x.clone().detach().requires_grad_(True)
            
            # Before feeding into the model: normalize
            x_n = (x - self.mean_tensor)/self.std_tensor
            
            x_out = self.model(x_n)
            loss = self.attack_criterion(x_out, lbl)

            with torch.no_grad():
                grad, = torch.autograd.grad(loss, [x])
                
                # Take one step
                l = len(x.shape) - 1
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, *([1]*l))
                scaled_grad = grad / (grad_norm + 1e-10)
                x = x - scaled_grad * self.step_size

                # Project
                pert = x - img
                pert = pert.renorm(p=2, dim=0, maxnorm=self.epsilon)
                x = torch.clamp(img + pert, self.img_min, self.img_max)
                
        if self.return_normalized:
            # Normalize before output
            x_n = (x - self.mean_tensor)/self.std_tensor
            return x_n
        else:
            return x


class PGD_Attack_Linf():
    def __init__(self, model, attack_criterion, 
                num_channels, input_shape, mean, std,
                step_size=0.1, epsilon=1., iterations=100,
                return_normalized=True, img_min=0, img_max=1, cuda=True):
        """
        PGD L infinity attack
        
        Resources for reference:
            https://github.com/MadryLab/robustness/blob/master/robustness/attacker.py
            https://github.com/MadryLab/robustness/blob/master/robustness/attack_steps.py
        """
        self.model = model
        # Model should be in evaluation mode
        self.model = self.model.eval()

        self.attack_criterion = attack_criterion
        self.cuda = cuda
        
        self.mean_tensor = torch.ones(1, num_channels, *input_shape)
        for idx in range(num_channels):
            self.mean_tensor[:,idx] *= mean[idx]
        if cuda:
            self.mean_tensor = self.mean_tensor.cuda()

        self.std_tensor = torch.ones(1, num_channels, *input_shape)
        for idx in range(num_channels):
            self.std_tensor[:,idx] *= std[idx]
        if cuda:
            self.std_tensor = self.std_tensor.cuda()
        
        self.step_size = step_size 
        self.epsilon = epsilon
        self.iterations = iterations
        self.return_normalized = return_normalized
        self.img_min = img_min 
        self.img_max = img_max
        
    def run(self, orig_img, lbl):
        # Unnormalize
        img = orig_img * self.std_tensor + self.mean_tensor
        
        x = img.clone().detach().requires_grad_(True)
        
        for i in range(self.iterations):
            x = x.clone().detach().requires_grad_(True)
            
            # Before feeding into the model: normalize
            x_n = (x - self.mean_tensor)/self.std_tensor
            
            x_out = self.model(x_n)
            loss = self.attack_criterion(x_out, lbl)
            
            with torch.no_grad():
                grad, = torch.autograd.grad(loss, [x])
                
                # Take one step
                step = torch.sign(grad) * self.step_size
                x = x - step

                # Project
                pert = x - img
                pert = torch.clamp(pert, -self.epsilon, self.epsilon)
                x = torch.clamp(img + pert, self.img_min, self.img_max)
        
        if self.return_normalized:
            # Normalize before output
            x_n = (x - self.mean_tensor)/self.std_tensor
            return x_n
        else:
            return x