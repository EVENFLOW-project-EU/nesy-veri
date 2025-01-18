from auto_LiRPA.operators.base import *

from auto_LiRPA.operators.reduce import BoundReduceMax

# The `option != 'complex'` case is not used in the auto_LiRPA main paper.
class CustomBoundSoftmax(Bound):
    
    def __init__(self, attr=None, inputs=None, output_index=None, options={}):
        super().__init__(attr, inputs, output_index, options)
        self.eps = 1e-12  # Numerical stability constant
        self.options = options
        self.option = 'complex'
        self.max_denom = 30.0  # Stability threshold


    def handle_shape(self, x):
        """
        Handle input shape transformations for consistent bound computation.
        
        Args:
            x (Tensor): Input tensor of shape [..., n_classes]
            
        Returns:
            Tuple[Tensor, tuple]: Reshaped tensor and original shape
        """
        orig_shape = x.shape
        if len(x.shape) > 2:
            x = x.reshape(-1, x.shape[-1])
        return x, orig_shape

    def restore_shape(self, x, orig_shape):
        """
        Restore tensor to original shape after bound computation.
        
        Args:
            x (Tensor): Processed tensor
            orig_shape (tuple): Original input shape
            
        Returns:
            Tensor: Reshaped output matching input dimensions
        """
        if len(orig_shape) > 2:
            x = x.reshape(orig_shape)
        return x

    def forward(self, x):
        """
        Shape-aware forward pass with stable softmax computation.
        
        Args:
            x (Tensor): Input logits
            
        Returns:
            Tensor: Normalized probabilities
        """
        x_val, orig_shape = self.handle_shape(x)
        max_x = torch.max(x_val, dim=-1, keepdim=True)[0]
        exp_x = torch.exp(x_val - max_x)
        softmax_out = exp_x / (torch.sum(exp_x, dim=-1, keepdim=True) + self.eps)
        return self.restore_shape(softmax_out, orig_shape)

    def interval_propagate(self, *v):
        """
        Propagate intervals using LSE bounds with shape preservation.
        
        Args:
            *v: Input intervals (h_L, h_U)
            
        Returns:
            tuple: Normalized probability bounds
        """
        h_L, h_U = v[0]
        h_L, orig_shape = self.handle_shape(h_L)
        h_U = self.handle_shape(h_U)[0]
        
        # Compute stable bounds
        max_U = torch.max(h_U, dim=-1, keepdim=True)[0]
        h_L_stable = h_L - max_U
        h_U_stable = h_U - max_U
        
        exp_L = torch.exp(torch.clamp(h_L_stable, max=self.max_denom))
        exp_U = torch.exp(torch.clamp(h_U_stable, max=self.max_denom))
        
        # Initialize bounds
        n_classes = h_L.shape[-1]
        device = h_L.device
        lower = torch.zeros_like(h_L)
        upper = torch.ones_like(h_U)
        
        # Class-wise bound computation
        for i in range(n_classes):
            denom_L = torch.sum(exp_U, dim=-1, keepdim=True) - exp_U[..., i:i+1] + exp_L[..., i:i+1]
            denom_U = torch.sum(exp_L, dim=-1, keepdim=True) - exp_L[..., i:i+1] + exp_U[..., i:i+1]
            
            lower[..., i:i+1] = exp_L[..., i:i+1] / (denom_L + self.eps)
            upper[..., i:i+1] = exp_U[..., i:i+1] / (denom_U + self.eps)
            
        # Normalize and restore shape
        lower = self.restore_shape(lower, orig_shape)
        upper = self.restore_shape(upper, orig_shape)
        
        return lower, upper

    def bound_backward(self, last_lA, last_uA, *x, **kwargs):
        """
        Compute linear relaxation for backward bound propagation (CROWN).
        
        Args:
            last_lA: Last layer's lower bound coefficients
            last_uA: Last layer's upper bound coefficients
            *args: Additional arguments
            
        Returns:
            Tuple: Updated bounds for backward propagation
        """
        if last_lA is None and last_uA is None:
            return None, None, None
            
        # Get input bounds from previous layer
        prev_lb = self.inputs[0].lower
        prev_ub = self.inputs[0].upper
        
        def _get_relaxation_slopes(lb, ub):
            """Compute slopes for linear relaxation."""
            if lb is None and ub is None:
                return None
            diff = ub - lb
            mask = diff > 0
            slopes = torch.zeros_like(diff)
            slopes[mask] = (torch.exp(ub[mask]) - torch.exp(lb[mask])) / diff[mask]
            return slopes
            
        # Compute slopes for relaxation
        slopes = _get_relaxation_slopes(prev_lb, prev_ub)
        
        # Initialize A matrices
        lA = None if last_lA is None else last_lA.clone()
        uA = None if last_uA is None else last_uA.clone()
        
        if lA is not None and slopes is not None:
            # Lower bound relaxation
            lA = lA * slopes.unsqueeze(0)
            
        if uA is not None and slopes is not None:
            # Upper bound relaxation
            uA = uA * slopes.unsqueeze(0)
            
        return [(lA, uA)], 0, 0
    

    def bound_forward(self, dim_in: int, *x: Tuple[torch.Tensor, torch.Tensor]) -> LinearBound:
        """Compute forward bounds using CROWN method.
        
        Implements forward bound propagation for hybrid verification.
        
        Args:
            dim_in: Input dimension
            *x: Input bounds from previous layer
            
        Returns:
            LinearBound object containing bound parameters
        """
        # Extract input bounds
        h_L, h_U = x[0]
        batch_size = h_L.shape[0]
        
        # Compute LSE bounds
        # h_max = torch.max(h_U, dim=-1, keepdim=True)[0]
        # sum_exp_U = torch.sum(torch.exp(h_U - h_max), dim=-1, keepdim=True)
        # sum_exp_L = torch.sum(torch.exp(h_L - h_max), dim=-1, keepdim=True)
        
        # Compute slopes for linear relaxation
        diff = h_U - h_L
        mask = diff < 1e-6
        slope = torch.where(mask,
                          torch.exp(h_L),
                          (torch.exp(h_U) - torch.exp(h_L)) / diff)
        
        # Initialize matrices for linear bounds
        weight = torch.eye(dim_in, device=h_L.device).unsqueeze(0).repeat(batch_size, 1, 1)
        bias = torch.zeros(batch_size, dim_in, device=h_L.device)
        
        # Apply slope to weight matrix
        weight = weight * slope.unsqueeze(-2)
        
        return LinearBound(weight, bias, weight, bias, h_L, h_U)
    

class CustomConcat(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.axis = attr['axis']
    
    def forward(self, *x):
        assert self.axis == int(self.axis)
        return torch.cat(x, dim=self.axis)

    def interval_propagate(self, *v):
        lower_bounds, upper_bounds = zip(*v)
        return torch.cat(lower_bounds, dim=self.axis), torch.cat(upper_bounds, dim=self.axis)