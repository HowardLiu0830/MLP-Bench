import Lake
open Lake DSL

/-
The only Lean project this benchmark needs.

Every task file imports Mathlib and nothing else: the released statements have their
definitions inlined and carry no dependency on any other library. So this package declares
no library of its own -- it exists to pin a Mathlib revision and to give `lake env lean` a
project to resolve.

`autoImplicit false` matches the environment the statements were written and verified in.
Turning it on would make some of them elaborate that should not.
-/
package «mlpbench» where
  leanOptions := #[⟨`autoImplicit, false⟩]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.27.0-rc1"
