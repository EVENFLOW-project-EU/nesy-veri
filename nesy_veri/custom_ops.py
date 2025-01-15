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

    def bound_backward(self, last_lA, last_uA, *x, start_node=None, **kwargs):
        """
        Compute backward bounds with proper shape handling.
        
        Args:
            last_lA: Lower bound coefficients from previous layer
            last_uA: Upper bound coefficients from previous layer
            *x: Input tensors including bounds
            start_node: Starting node for bound computation
            **kwargs: Additional keyword arguments
            
        Returns:
            tuple: Updated bounds and bias terms
        """
        if last_lA is None and last_uA is None:
            return None, 0, 0
            
        # Handle input shapes
        x_L, x_U = x[0][0], x[0][1]
        x_L, orig_shape = self.handle_shape(x_L)
        x_U = self.handle_shape(x_U)[0]
        
        # Compute relaxation at multiple reference points
        alpha_points = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=x_L.device)
        ref_points = [x_L * (1 - a) + x_U * a for a in alpha_points]
        
        def get_bound_parameters(x_ref):
            """Compute bound parameters at reference point."""
            max_ref = torch.max(x_ref, dim=-1, keepdim=True)[0]
            x_shifted = x_ref - max_ref
            exp_ref = torch.exp(torch.clamp(x_shifted, max=self.max_denom))
            sum_exp = torch.sum(exp_ref, dim=-1, keepdim=True)
            return exp_ref / (sum_exp + self.eps)
            
        # Get slopes at reference points
        slopes = [get_bound_parameters(ref) for ref in ref_points]
        
        # Compute bounds using optimal slopes
        if last_lA is not None:
            lA = torch.where(
                last_lA >= 0,
                torch.min(torch.stack([s * last_lA for s in slopes]), dim=0)[0],
                torch.max(torch.stack([s * last_lA for s in slopes]), dim=0)[0]
            )
        else:
            lA = None
            
        if last_uA is not None:
            uA = torch.where(
                last_uA >= 0,
                torch.max(torch.stack([s * last_uA for s in slopes]), dim=0)[0],
                torch.min(torch.stack([s * last_uA for s in slopes]), dim=0)[0]
            )
        else:
            uA = None
            
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
        h_max = torch.max(h_U, dim=-1, keepdim=True)[0]
        sum_exp_U = torch.sum(torch.exp(h_U - h_max), dim=-1, keepdim=True)
        sum_exp_L = torch.sum(torch.exp(h_L - h_max), dim=-1, keepdim=True)
        
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