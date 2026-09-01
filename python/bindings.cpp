/**
 * placecell Python bindings (nanobind).
 */
#include <nanobind/nanobind.h>

#include <placecell/placecell.h>

namespace nb = nanobind;

NB_MODULE(_placecell, m)
{
    m.doc() = "Keyframe lifecycle management for VSLAM and 3D reconstruction";

    nb::class_<placecell::PlaceCell>(m, "PlaceCell")
        .def(nb::init<>());
}
