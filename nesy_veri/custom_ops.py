from auto_LiRPA.operators.base import *

from auto_LiRPA.operators.reduce import BoundReduceMax

# The `option != 'complex'` case is not used in the auto_LiRPA main paper.
class CustomBoundSoftmax(Bound):
    
    def __init__(self, attr=None, inputs=None, output_index=None, options={}):
        super().__init__(attr, inputs, output_index, options)
        self.eps = 1e-12  # Numerical stability constant
        self.options = options
        self.axis = attr['axis']
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
        
        assert torch.all(lower) <= 1 and torch.all(lower) >= 0
        assert torch.all(upper) <= 1 and torch.all(upper) >= 0 
        return lower, upper

    def bound_backward(self, last_lA, last_uA, *x, **kwargs):
        """
        Compute CROWN backward bounds for LSE Softmax using matrix-based approach.
        
        Args:
            last_lA: Last layer's lower bound linear coefficients
            last_uA: Last layer's upper bound linear coefficients
            x: Input bounds from previous layer
            
        Returns:
            tuple: (A_matrices, lbias, ubias) for bound propagation
        """
        # Extract pre-activation bounds
        h_L, h_U = x[0].lower, x[0].upper
        default_shape = x[0].output_shape
        # Default to conservative bounds if real bounds unavailable
        if h_L is None:
            h_L = torch.full(default_shape, -1e9, device=self.device)
        if h_U is None:
            h_U = torch.full(default_shape, 1e9, device=self.device)
        
        batch_size = h_L.shape[0] if h_L is not None and len(h_L.shape) > 1 else 1
        
        def _compute_relaxation_parameters(h_L, h_U, default_shape=None):
            """Compute optimal relaxation parameters for linear bounds."""
                
            exp_l = torch.exp(h_L)
            exp_u = torch.exp(h_U)
            diff = h_U - h_L
            
            # Prevent numerical instability
            mask = diff < 1e-6
            diff = torch.where(mask, torch.ones_like(diff)*1e-6, diff)
            
            # Compute slope candidates
            k1 = torch.log((exp_u - exp_l) / diff)
            k2 = h_L + 1
            
            # Select optimal slope while ensuring numerical stability
            k = torch.min(k1, k2)
            slope_lower = torch.exp(k)
            slope_upper = (exp_u - exp_l) / diff
            
            return slope_lower, slope_upper, exp_l, exp_u

        def _process_A_matrix(A_matrix, slopes_lower, slopes_upper, bias_lower, bias_upper):
            """Process A matrix with computed slopes for bound computation."""
            if A_matrix is None:
                return None, 0
                
            # Select slopes based on A matrix sign
            pos_mask = A_matrix > 0
            neg_mask = A_matrix <= 0
            
            # Compute final slopes
            slopes = torch.zeros_like(A_matrix)
            slopes = torch.where(pos_mask, slopes_upper, slopes)
            slopes = torch.where(neg_mask, slopes_lower, slopes)
            
            # Compute bias terms
            bias = torch.zeros_like(A_matrix)
            bias = torch.where(pos_mask, bias_lower, bias)
            bias = torch.where(neg_mask, bias_upper, bias)
            
            # Final A matrix and bias computation
            new_A = A_matrix * slopes
            new_bias = (A_matrix * bias).sum(dim=-1)
            
            return new_A, new_bias

        # Compute relaxation parameters
        slopes_l, slopes_u, exp_l, exp_u = _compute_relaxation_parameters(h_L, h_U, default_shape)
        
        # Compute bias terms for relaxation
        bias_l = exp_l - slopes_l * h_L
        bias_u = exp_u - slopes_u * h_U
        
        # Process bounds for lower and upper A matrices
        lA, lbias = _process_A_matrix(last_lA, slopes_l, slopes_u, bias_l, bias_u)
        uA, ubias = _process_A_matrix(last_uA, slopes_l, slopes_u, bias_l, bias_u)
        
        # Return processed bounds
        return [(lA, uA)], lbias, ubias

    
    def bound_forward(self, dim_in: int, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Implements CROWN+IBP forward bound propagation.
        
        Args:
            dim_in: Input dimension
            x: Input bound information
            
        Returns:
            Tuple containing bounds and gradient information
        """
        def _bound_oneside(x_lb: torch.Tensor, x_ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # Handle potentially missing bounds
            if x_lb is None or x_ub is None:
                raise ValueError("Forward bound propagation requires valid bounds")

            # Compute exp bounds with numerical stability
            max_val = torch.max(x_ub)
            exp_l = torch.exp(x_lb - max_val)
            exp_u = torch.exp(x_ub - max_val)

            # Reference point computation
            ref_point = (x_lb + x_ub) / 2
            exp_ref = torch.exp(ref_point - max_val)
            
            # Sum of exponentials at reference point
            sum_exp_ref = exp_ref.sum(dim=1, keepdim=True)

            # Gradient computation
            grad = exp_ref * (1 - exp_ref / (sum_exp_ref + self.epsilon))

            # Bound computation with stability
            lb = exp_l / (exp_u.sum(dim=1, keepdim=True) + self.epsilon)
            ub = exp_u / (exp_l.sum(dim=1, keepdim=True) + self.epsilon)

            return lb, ub, grad

        # Get bounds from input
        lb = x.lower if hasattr(x, 'lower') else None
        ub = x.upper if hasattr(x, 'upper') else None

        return _bound_oneside(lb, ub)
    

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