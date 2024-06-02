import torch
import qpth

# Define the quadratic term Q (positive definite matrix)
Q = torch.tensor([[4., 1.],
                  [1., 2.]], requires_grad=True)

# Define the linear term p
p = torch.tensor([1., 1.], requires_grad=True)

# Create dummy inequality constraints (large h to ensure no constraint)
G = torch.zeros(1, 2)  # No actual constraints
h = torch.tensor([1e10])  # Large value to effectively ignore G
# G = None
# H = None

# Define the equality constraints A and b
A = torch.tensor([[1., 1.]], requires_grad=True)
b = torch.tensor([1.], requires_grad=True)

# Use QPFunction to solve the quadratic program
qp = qpth.qp.QPFunction()
solution = qp(Q, p, G, h, A, b)

print("Solution:", solution)