import cutlass.cute as cute


@cute.jit
def ssa_to_indexable(ssa_value: cute.TensorSSA, dtype) -> cute.Numeric:
    """
    Convert SSA form to indexable non-SSA form.

    Workaround for lack of gather support: SSA values cannot be used directly
    as indices in tensor loads. This converts SSA → fragment → scalar for indexing.
    """
    frag = cute.make_fragment(1, dtype)
    frag.store(ssa_value)
    return frag[0]


@cute.jit
def result_to_ssa(value: cute.Numeric, dtype) -> cute.TensorSSA:
    """
    Convert non-SSA result back to SSA form.

    After performing operations with non-SSA values (like indexed loads),
    convert the result back to SSA form for further computation.
    """
    frag = cute.make_fragment(1, dtype)
    frag[0] = value
    return frag.load()
