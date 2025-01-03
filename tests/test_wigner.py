import torch
from spherical.cuda._backend import _C
from spherical.cuda._naive_impl import wignerD
from tqdm import tqdm

euler = torch.rand(1, 3).cuda()

# maximum degree without overflow: 17
# for degree in range(1, 18):
#     real, imag = _C.wignerD_fwd(euler, degree)
#     real_old, imag_old = _C.wignerD_fwd_old(euler, degree)

#     # _real, _imag = wignerD(euler.cpu().numpy(), degree)
#     # _real, _imag = (
#     #     torch.from_numpy(_real).cuda().float(),
#     #     torch.from_numpy(_imag).cuda().float(),
#     # )
#     if (
#         torch.isnan(real).any()
#         or torch.isnan(imag).any()
#         or torch.isnan(real_old).any()
#         or torch.isnan(imag_old).any()
#     ):
#         print(degree)
#         print(f"nan in real: {torch.isnan(real).sum().item()}")
#         print(f"nan in imag: {torch.isnan(imag).sum().item()}")
#         print(f"nan in real_old: {torch.isnan(real_old).sum().item()}")
#         print(f"nan in imag_old: {torch.isnan(imag_old).sum().item()}")
#     else:
#         torch.testing.assert_close(real, real_old)
#         torch.testing.assert_close(imag, -imag_old)
#         print(f"degree {degree} passed")

# test speed
for degree in range(1, 18):
    pbar = tqdm(range(1000))
    pbar.set_description(f"degree {degree}, new")
    for _ in pbar:
        real, imag = _C.wignerD_fwd(euler, degree)

    pbar = tqdm(range(1000))
    pbar.set_description(f"degree {degree}, old")
    for _ in pbar:
        real_old, imag_old = _C.wignerD_fwd_old(euler, degree)
