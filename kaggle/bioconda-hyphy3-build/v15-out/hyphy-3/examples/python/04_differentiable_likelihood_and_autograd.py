#!/usr/bin/env python3
"""
Example 4: Differentiable Phylogenetics & Inside-Outside Adjoint Gradients
Demonstrates analytical gradient computation d ln L / d t_b and validation against finite differences.
"""

import sys
import numpy as np
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "python"))

import hyphy3 as hp

def main():
    data_dir = repo_root / "benchmarks" / "data"
    aln_path = str(data_dir / "cd2.fna")
    # Use the fitted tree: gradients at zero-length branches are degenerate
    tree_path = str(data_dir / "cd2_fitted.nwk")

    aln = hp.Alignment.load(aln_path)
    tree = hp.Tree.from_newick_file(tree_path)

    # 1. Baseline MG94 parameters
    params = hp.MG94Parameters()
    params.alpha = 1.0
    params.beta = 0.5
    params.theta_AC = 0.8
    params.theta_AT = 0.4
    params.theta_CG = 0.5
    params.theta_CT = 1.2
    params.theta_GT = 0.3

    # Extract initial branch lengths from tree
    bls = [node.branch_length for node in tree.nodes]

    # 2. Compute exact analytical Inside-Outside gradients in a single O(B) pass
    print("Computing exact analytical Inside-Outside gradients...")
    log_l, analytical_grads = hp.compute_branch_length_gradients(tree, aln, params, bls)
    print(f"Initial Tree Log-Likelihood: {log_l:.4f}")

    # 3. Verify against numerical central finite differences
    print("\nComparing analytical adjoint gradients against numerical finite differences:")
    print(f"{'Branch ID':<10} {'Node Name':<20} {'Analytical':<15} {'Finite Diff':<15} {'Rel Error':<12}")
    print("-" * 75)

    eps = 1e-6
    max_rel_err = 0.0
    for node in tree.nodes:
        if node.id == tree.root_id:
            continue
        nid = node.id
        
        # Central difference: (ln L(t + eps) - ln L(t - eps)) / (2 * eps)
        bl_plus = list(bls)
        bl_plus[nid] += eps
        ll_plus, _ = hp.compute_branch_length_gradients(tree, aln, params, bl_plus)

        bl_minus = list(bls)
        bl_minus[nid] -= eps
        ll_minus, _ = hp.compute_branch_length_gradients(tree, aln, params, bl_minus)

        fd_grad = (ll_plus - ll_minus) / (2.0 * eps)
        ana_grad = analytical_grads[nid]
        rel_err = abs(ana_grad - fd_grad) / (abs(ana_grad) + 1e-8)
        max_rel_err = max(max_rel_err, rel_err)

        print(f"{nid:<10} {node.name:<20} {ana_grad:<15.6f} {fd_grad:<15.6f} {rel_err:<12.2e}")

    if max_rel_err > 1e-4:
        print(f"\nGradient check FAILED: max relative error {max_rel_err:.2e}")
        sys.exit(1)
    print(f"\nAll analytical gradients match finite differences (max relative error {max_rel_err:.2e}).")

if __name__ == "__main__":
    main()
