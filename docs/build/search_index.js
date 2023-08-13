var documenterSearchIndex = {"docs":
[{"location":"index.html","page":"Home","title":"Home","text":"CurrentModule = InvariantPointAttention","category":"page"},{"location":"index.html#InvariantPointAttention","page":"Home","title":"InvariantPointAttention","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"Documentation for InvariantPointAttention.","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"Modules = [InvariantPointAttention]","category":"page"},{"location":"index.html#InvariantPointAttention.BackboneUpdate","page":"Home","title":"InvariantPointAttention.BackboneUpdate","text":"Projects the frame embedding => 6, and uses this to transform the input frames.\n\n\n\n\n\n","category":"type"},{"location":"index.html#InvariantPointAttention.IPA","page":"Home","title":"InvariantPointAttention.IPA","text":"Invariant Point Attention\n\n\n\n\n\n","category":"type"},{"location":"index.html#InvariantPointAttention.IPAStructureModuleLayer","page":"Home","title":"InvariantPointAttention.IPAStructureModuleLayer","text":"Partial Structure Module - single layer - from AF2. Not a faithful repro, and doesn't include the losses etc.\n\n\n\n\n\n","category":"type"},{"location":"index.html#InvariantPointAttention.IPA_settings-Tuple{Any}","page":"Home","title":"InvariantPointAttention.IPA_settings","text":"Returns a tuple of the IPA settings, with defaults for everything except dims. This can be passed to the IPA and IPAStructureModuleLayer.\n\n\n\n\n\n","category":"method"},{"location":"index.html#InvariantPointAttention.T_R3-Tuple{Any, Any, Any}","page":"Home","title":"InvariantPointAttention.T_R3","text":"Applies the SE3 transformations T = (rot,trans) ∈ SE(E3)^N to N batches of m points in R3, i.e., mat ∈ R^(3 x m x N) ↦ T(mat) ∈ R^(3 x m x N). Note here that rotations here are represented in matrix form. \n\n\n\n\n\n","category":"method"},{"location":"index.html#InvariantPointAttention.T_R3_inv-Tuple{Any, Any, Any}","page":"Home","title":"InvariantPointAttention.T_R3_inv","text":"Applys the group inverse of the SE3 transformations T = (rot,trans) ∈ SE(3)^N to N batches of m points in R3, i.e., mat ∈ R^(3 x m x N) ↦ T^(-1)(mat) ∈ R^(3 x m x N) such that T(T^-1(mat)) = mat = T^-1(T(mat)).  Note here that rotations here are represented in matrix form.  \n\n\n\n\n\n","category":"method"},{"location":"index.html#InvariantPointAttention.T_T-Tuple{Any, Any}","page":"Home","title":"InvariantPointAttention.T_T","text":"Returns the composition of two SE(3) transformations T1 and T2. Note that if T1 = (R1,t1), and T2 = (R2,t2) then T1T2 = (R1R2, R1*t2 + t1). T here is stored as a tuple (R,t).\n\n\n\n\n\n","category":"method"},{"location":"index.html#InvariantPointAttention.bcds2quats-Tuple{AbstractMatrix{<:Real}}","page":"Home","title":"InvariantPointAttention.bcds2quats","text":"Creates a quaternion (as a vector) from a triplet of values (pirated from Diffusions.jl)\n\n\n\n\n\n","category":"method"},{"location":"index.html#InvariantPointAttention.get_rotation-Tuple{Any, Any}","page":"Home","title":"InvariantPointAttention.get_rotation","text":"Gets N random rotation matrices formatted as an array of size 3x3xN. \n\n\n\n\n\n","category":"method"},{"location":"index.html#InvariantPointAttention.get_translation-Tuple{Any, Any}","page":"Home","title":"InvariantPointAttention.get_translation","text":"Gets N random translations formatted as an array of size 3x1xN (for purposes of broadcasting to arrays of size 3 x m x N)\n\n\n\n\n\n","category":"method"},{"location":"index.html#InvariantPointAttention.rotmatrix_from_quat-Tuple{Any}","page":"Home","title":"InvariantPointAttention.rotmatrix_from_quat","text":"Returns the rotation matrix form of N flat quaternions. \n\n\n\n\n\n","category":"method"},{"location":"index.html#InvariantPointAttention.update_frame-Tuple{Any, Any}","page":"Home","title":"InvariantPointAttention.update_frame","text":"Takes a 6-dim vec and maps to a rotation matrix and translation vector, which is then applied to the input frames.\n\n\n\n\n\n","category":"method"}]
}