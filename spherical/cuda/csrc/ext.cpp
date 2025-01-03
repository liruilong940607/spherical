#include "bindings.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wignerD_fwd", &wignerD_fwd);
    m.def("wignerD_fwd_old", &wignerD_fwd_old);
}
