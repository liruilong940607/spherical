import torch
from spherical.cuda._backend import _C
from spherical.cuda._native_impl import wignerD

euler = torch.rand(1, 3).cuda()
real, imag = _C.wignerD_fwd(euler, 1)

_real, _imag = wignerD(euler.cpu().numpy(), 1)
_real, _imag = torch.from_numpy(_real).cuda().float(), torch.from_numpy(_imag).cuda().float()

print(torch.allclose(real, _real))
print(torch.allclose(imag, _imag))


