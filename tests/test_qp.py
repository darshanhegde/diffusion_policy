import unittest
import torch
import qpth

class TestQPTH(unittest.TestCase):
    def setUp(self):
        self.n = 10  # Number of variables
        self.m = 5   # Number of inequality constraints

        # Random QP problem
        self.Q = torch.rand(self.n, self.n)
        self.Q = self.Q @ self.Q.t()  # Make Q positive semi-definite
        self.p = torch.rand(self.n)
        self.G = torch.rand(self.m, self.n)
        self.h = torch.rand(self.m)

        # Solver
        self.solver = qpth.qp.QPFunction(verbose=-1)

    def test_inequality_constraints(self):
        # Solve the QP problem
        Q = self.Q.double().unsqueeze(0)  # Add batch dimension
        p = self.p.double().unsqueeze(0)  # Add batch dimension
        G = self.G.double().unsqueeze(0)  # Add batch dimension
        h = self.h.double().unsqueeze(0)  # Add batch dimension
        A = torch.empty(0, self.n).double().unsqueeze(0)  # Add batch dimension
        b = torch.empty(0).double().unsqueeze(0)  # Add batch dimension

        x = self.solver(Q, p, G, h, A, b)

        # Remove batch dimension for verification
        x = x.squeeze(0)

        # Check if inequality constraints are satisfied
        Gx = torch.matmul(self.G.double(), x)
        satisfied = (Gx <= self.h.double()).all().item()

        self.assertTrue(satisfied, "Inequality constraints are not satisfied")

if __name__ == '__main__':
    unittest.main()