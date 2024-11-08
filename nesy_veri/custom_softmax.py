from auto_LiRPA.operators.base import *

# The `option != 'complex'` case is not used in the auto_LiRPA main paper.
class CustomBoundSoftmax(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.axis = attr['axis']
        self.option = 'softmax'
        self.max_input = 30

    def forward(self, x):
        assert self.axis == int(self.axis)
        return F.softmax(x, dim=self.axis)

    def interval_propagate(self, *v):
        assert self.option != 'complex'
        assert self.perturbed
        h_L, h_U = v[0]
        shift = h_U.max(dim=self.axis, keepdim=True).values
        exp_L, exp_U = torch.exp(h_L - shift), torch.exp(h_U - shift)
        lower = exp_L / (torch.sum(exp_U, dim=self.axis, keepdim=True) - exp_U + exp_L + epsilon)
        upper = exp_U / (torch.sum(exp_L, dim=self.axis, keepdim=True) - exp_L + exp_U + epsilon)
        return lower, upper