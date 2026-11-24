// Shared declarations for the _hyphy3 nanobind module.
//
// Each bindings_<group>.cpp TU defines one register_<group>() function; the
// thin bindings.cpp module entry point calls them all in order. Splitting the
// registrars keeps any single TU from parsing every analysis header at once
// (the old monolith was the peak-RSS offender in the bindings build).
//
// The nanobind stl/eigen caster headers are included here so every bindings TU
// gets a uniform, sufficient include set; the cost is small next to the
// analysis headers each TU pulls in.

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/array.h>
#include <nanobind/eigen/dense.h>

namespace hyphy::bindings {

void register_core(nanobind::module_ &);
void register_fel(nanobind::module_ &);
void register_meme(nanobind::module_ &);
void register_busted(nanobind::module_ &);
void register_absrel(nanobind::module_ &);
void register_relax(nanobind::module_ &);
void register_fubar(nanobind::module_ &);
void register_prime(nanobind::module_ &);
void register_gard(nanobind::module_ &);
void register_slac(nanobind::module_ &);
void register_dfel(nanobind::module_ &);
void register_ohmic(nanobind::module_ &);

} // namespace hyphy::bindings
