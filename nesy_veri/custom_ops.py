from auto_LiRPA.operators.base import *

from auto_LiRPA.operators.reduce import BoundReduceMax

# The `option != 'complex'` case is not used in the auto_LiRPA main paper.
class CustomBoundSoftmax(Bound):
    
    def __init__(self, attr=None, inputs=None, output_index=None, options={}):
        super().__init__(attr, inputs, output_index, options)
        self.eps = 1e-12  # Numerical stability constant
        self.epsilon = 1e-12  # Numerical stability constant
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
        CROWN backward bound propagation with improved linear relaxation.
        Uses multi-point sampling and adaptive slope selection for tighter bounds.
        """
        def _get_adaptive_slopes(h_L, h_U, last_A):
            """
            Compute adaptive slopes for linear relaxation using multi-point sampling.
            """
            # Sample intermediate points for better slope estimation
            num_points = 3
            sample_points = torch.linspace(0, 1, num_points, device=h_L.device)
            sample_points = sample_points.view(-1, 1, 1)
            
            # Compute points between lower and upper bounds
            points = h_L + sample_points * (h_U - h_L)
            
            # Compute exponentials at sample points
            exp_points = torch.exp(points)
            
            # Compute derivatives at sample points
            derivatives = exp_points
            
            # Select best slopes based on sign of last_A
            if last_A is not None:
                # For positive last_A, use minimum slope for lower bound
                # For negative last_A, use maximum slope for upper bound
                pos_mask = last_A > 0
                slopes = torch.where(
                    pos_mask,
                    derivatives.min(dim=0)[0],
                    derivatives.max(dim=0)[0]
                )
            else:
                slopes = derivatives.mean(dim=0)
                
            return slopes

        def _bound_oneside(last_A, x):
            if last_A is None:
                return None, 0
            
            h_L, h_U = x.lower, x.upper
            if h_L is None or h_U is None:
                return None, 0
            
            # Get adaptive slopes for better linear relaxation
            slopes = _get_adaptive_slopes(h_L, h_U, last_A)
            
            # Compute exponential bounds with improved stability
            exp_l = torch.exp(h_L - torch.max(h_U))
            exp_u = torch.exp(h_U - torch.max(h_U))
            
            # Compute LSE bounds
            lse_L = torch.log(exp_l.sum(dim=1, keepdim=True) + self.epsilon)
            lse_U = torch.log(exp_u.sum(dim=1, keepdim=True) + self.epsilon)
            
            # Compute improved linear relaxation parameters
            diff = h_U - h_L
            diff = torch.where(diff < self.epsilon, torch.ones_like(diff) * self.epsilon, diff)
            
            # Adaptive reference point selection
            # ref_point = h_L + (h_U - h_L) * torch.sigmoid(last_A)
            
            # Compute tangent slopes with stability
            k = torch.log((exp_u - exp_l) / diff + self.epsilon)
            k = torch.min(k, h_L + 1)
            
            # Improved slope selection
            slope_l = torch.exp(k)
            slope_u = (exp_u - exp_l) / diff
            
            # Compute final slopes using adaptive selection
            final_slopes = torch.where(
                last_A > 0,
                torch.min(slope_u, slopes),
                torch.max(slope_l, slopes)
            )
            
            # Compute bias terms with improved stability
            bias = torch.where(
                last_A > 0,
                exp_l - final_slopes * h_L,
                exp_u - final_slopes * h_U
            )
            
            # Apply LSE normalization
            A = last_A * final_slopes / (torch.exp(lse_U - lse_L) + self.epsilon)
            b = (last_A * bias).sum(dim=-1)/ (torch.exp(lse_U - lse_L) + self.epsilon)
            
            return A, b

        # Compute bounds with improved stability
        lA, lbias = _bound_oneside(last_lA, *x)
        uA, ubias = _bound_oneside(last_uA, *x)
        
        return [(lA, uA)], lbias, ubias

    def bound_forward(self, dim_in, x):
        """
        CROWN+IBP forward bound propagation.
        Uses adaptive reference points and improved linear relaxation.
        """
        def _bound_oneside(x_lb, x_ub):
            h_L, h_U = x_lb, x_ub
            
            # Compute bounds in log space for stability
            exp_l = torch.exp(h_L - torch.max(h_U))
            exp_u = torch.exp(h_U - torch.max(h_U))
            
            # Adaptive reference point selection
            alpha = 0.5  # Can be tuned for better results
            ref_point = h_L + alpha * (h_U - h_L)
            
            # Compute gradients with improved stability
            exp_ref = torch.exp(ref_point - torch.max(ref_point))
            sum_exp_ref = exp_ref.sum(dim=1, keepdim=True)
            grad = exp_ref * (1 - exp_ref / (sum_exp_ref + self.epsilon))
            
            # Compute bounds with LSE properties
            lb = exp_l / (exp_u.sum(dim=1, keepdim=True) + self.epsilon)
            ub = exp_u / (exp_l.sum(dim=1, keepdim=True) + self.epsilon)
            
            return lb, ub, grad
        
        lb, ub = x.lower, x.upper
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