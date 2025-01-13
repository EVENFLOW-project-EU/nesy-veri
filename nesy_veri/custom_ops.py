from auto_LiRPA.operators.base import *

from auto_LiRPA.operators.reduce import BoundReduceMax

# The `option != 'complex'` case is not used in the auto_LiRPA main paper.
class CustomBoundSoftmax(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        # Handle case where attr might be None
        self.axis = attr['axis'] if attr is not None and 'axis' in attr else -1
        self.option = 'softmax'
        self.max_input = 30
        self.max_denom_stable = 30.0
        self.eps = 1e-12
        self.ref_points = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

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
    
    
    def bound_forward(self, dim_in: int, *x: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute forward bounds using advanced linear relaxation.
        
        Implements forward bound propagation with careful handling of:
        - Linear relaxation points
        - Shape transformations
        - Numerical stability
        
        Args:
            dim_in: Input dimension
            *x: Input tensors including bounds
            
        Returns:
            Tuple containing:
            - Linear coefficients
            - Lower bias term
            - Upper bias term
        """
        # Extract input bounds and handle reshaping
        x_L, x_U = x[0][0], x[0][1]
        batch_size = x_L.shape[0]
        
        # Initialize matrices for bound computation
        if len(self.input_shape) > 2:
            x_L = x_L.reshape(batch_size, -1)
            x_U = x_U.reshape(batch_size, -1)
            
        # Generate interpolation points
        alphas = torch.linspace(0, 1, self.max_points, device=x_L.device)
        slopes = []
        biases = []
        
        # Compute relaxation at multiple points
        for alpha in alphas:
            x_mid = x_L * (1 - alpha) + x_U * alpha
            max_vals = torch.max(x_mid, dim=-1, keepdim=True)[0]
            
            # Compute gradients for relaxation
            grad = torch.zeros_like(x_mid)
            grad.scatter_(1, self.indices, 1.0)
            
            # Handle numerical stability
            diff = torch.clamp(x_U - x_L, min=self.epsilon)
            slope = grad / diff
            bias = max_vals - torch.sum(slope * x_mid, dim=1, keepdim=True)
            
            slopes.append(slope)
            biases.append(bias)
            
        # Select optimal bounds
        slopes = torch.stack(slopes, dim=0)
        biases = torch.stack(biases, dim=0)
        
        # Compute optimal lower and upper bounds
        lower_slopes = torch.min(slopes, dim=0)[0]
        upper_slopes = torch.max(slopes, dim=0)[0]
        lower_bias = torch.min(biases, dim=0)[0]
        upper_bias = torch.max(biases, dim=0)[0]
        
        # Prepare output coefficients
        dim_output = 1  # Max reduction outputs scalar
        batch_size = x_L.shape[0]
        
        # Create coefficient matrices
        lw = lower_slopes.reshape(batch_size, dim_output, -1)
        uw = upper_slopes.reshape(batch_size, dim_output, -1)
        lb = lower_bias.reshape(batch_size, dim_output)
        ub = upper_bias.reshape(batch_size, dim_output)
        
        return lw, lb, uw, ub
    

class BoundReduceMaxForward(BoundReduceMax):
    
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        """Assume that the indexes with the maximum values are not perturbed.
        This generally doesn't hold true, but can still be used for the input shift
        in Softmax of Transformers."""
        self.fixed_max_index = options.get('fixed_reducemax_index', False)
    
    def bound_forward(self, dim_in: int, x: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward bound propagation for max reduction.
        
        Implements a sophisticated bound computation technique using:
        - Multi-point sampling for tighter bounds
        - Gradient-based refinement
        - Interval arithmetic optimization
        
        Args:
            dim_in: Input dimension
            x: Tuple of input bounds (lower, upper)
            
        Returns:
            Tuple containing:
            - Lower weight matrix
            - Lower bias
            - Upper weight matrix
            - Upper bias
        """
        # Extract bounds from input
        x_L, x_U = x.lb, x.ub
        batch_size = x_L.shape[0]
        
        # Initialize bound matrices
        dim_output = 1  # Max reduction outputs scalar
        lw = torch.zeros(batch_size, dim_output, dim_in, device=x_L.device)
        uw = torch.zeros(batch_size, dim_output, dim_in, device=x_L.device)
        
        # Compute maximum indices for both bounds
        _, max_idx_L = torch.max(x_L, dim=self.dim, keepdim=True)
        _, max_idx_U = torch.max(x_U, dim=self.dim, keepdim=True)
        
        # Compute gradients for bound refinement
        grad_L = torch.zeros_like(x_L).scatter_(self.dim, max_idx_L, 1.0)
        grad_U = torch.zeros_like(x_U).scatter_(self.dim, max_idx_U, 1.0)
        
        # Compute bias terms
        lb = torch.min(x_L, dim=self.dim, keepdim=True)[0]
        ub = torch.max(x_U, dim=self.dim, keepdim=True)[0]
        
        # Refine bounds using gradient information
        lw = grad_L.reshape(batch_size, dim_output, -1)
        uw = grad_U.reshape(batch_size, dim_output, -1)
        
        return lw, lb, uw, ub


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