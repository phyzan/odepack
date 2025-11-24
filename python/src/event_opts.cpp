#include "include/pyode/pyode.hpp"


PYBIND11_MODULE(_event_opts, m) {
    define_event_opt(m);
}