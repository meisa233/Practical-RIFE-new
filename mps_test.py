import torch
import timeit
def test_cpu():
    a_cpu = torch.rand(1000, device='cpu')
    b_cpu = torch.rand((1000, 1000), device='cpu')
    a_cpu @ b_cpu
def test_mps():
    a_mps = torch.rand(1000, device='mps')
    b_mps = torch.rand((1000, 1000), device='mps')
    a_mps @ b_mps

print('cpu', timeit.timeit(lambda: test_cpu(), number=1000))
print('mps', timeit.timeit(lambda: test_mps(), number=1000))