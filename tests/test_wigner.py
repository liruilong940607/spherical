import torch
from spherical.cuda._backend import _C
from spherical.cuda._naive_impl import wignerD

euler = torch.rand(1, 3).cuda()

# test correctness
# maximum degree supported: 16
for degree in range(1, 17):
    real, imag = _C.wignerD_fwd(euler, degree)
    real_torch, imag_torch = wignerD(euler.cpu().numpy(), degree)
    real_torch = torch.from_numpy(real_torch).cuda().float()
    imag_torch = torch.from_numpy(imag_torch).cuda().float()
    torch.testing.assert_close(real, real_torch, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(imag, imag_torch, atol=1e-3, rtol=1e-3)
