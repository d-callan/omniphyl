// _hyphy3 module entry point. All bound classes/functions are registered by
// the per-group TUs in this directory (see bindings_registry.hpp); this file
// only sequences the calls. Keep call order stable: core types first since
// other groups' signatures reference them.

#include "bindings_registry.hpp"

NB_MODULE(_hyphy3, m) {
    m.doc() =
        "HyPhy 3: Next-Generation High-Performance Molecular Evolution & "
        "Phylogenetics";

    hyphy::bindings::register_core(m);
    hyphy::bindings::register_fel(m);
    hyphy::bindings::register_meme(m);
    hyphy::bindings::register_busted(m);
    hyphy::bindings::register_absrel(m);
    hyphy::bindings::register_relax(m);
    hyphy::bindings::register_fubar(m);
    hyphy::bindings::register_prime(m);
    hyphy::bindings::register_gard(m);
    hyphy::bindings::register_slac(m);
    hyphy::bindings::register_dfel(m);
    hyphy::bindings::register_ohmic(m);
}
