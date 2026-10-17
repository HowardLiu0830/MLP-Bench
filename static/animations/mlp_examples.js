const hoeffding = String.raw`<p>Let \(X_1, \ldots, X_n\) be independent random variables taking values in \([0,1]\), with common mean \(\mu\). For every \(\varepsilon > 0\),</p><div class="formula">\[\begin{aligned}\Pr\!\left(\overline{X}_n \ge \mu + \varepsilon\right) &\le e^{-2n\varepsilon^2},\\\Pr\!\left(\overline{X}_n \le \mu - \varepsilon\right) &\le e^{-2n\varepsilon^2}.\end{aligned}\]</div>`;
const statement = `theorem example_theorem
    {Ω : Type*} [MeasurableSpace Ω]
    {n : ℕ} (hn : 0 < n)
    (P : Measure Ω) [IsProbabilityMeasure P]
    (X : Fin n → Ω → ℝ) (m ε : ℝ)
    (h_meas : ∀ i, Measurable (X i))
    (h_indep : iIndepFun X P)
    (h_bound : ∀ i, ∀ᵐ ω ∂P, X i ω ∈ Set.Icc (0 : ℝ) 1)
    (hm : ∀ i, ∫ ω, X i ω ∂P = m)
    (hε : 0 < ε) :
    P.real {ω | m + ε ≤ (n : ℝ)⁻¹ * ∑ i, X i ω}
      ≤ Real.exp (-2 * (n : ℝ) * ε ^ 2) ∧
    P.real {ω | (n : ℝ)⁻¹ * ∑ i, X i ω ≤ m - ε}
      ≤ Real.exp (-2 * (n : ℝ) * ε ^ 2) := by
  sorry`;
const theoremProof = `import Mathlib

open MeasureTheory ProbabilityTheory Real

theorem example_theorem
    {Ω : Type*} [MeasurableSpace Ω]
    {n : ℕ} (hn : 0 < n)
    (P : Measure Ω) [IsProbabilityMeasure P]
    (X : Fin n → Ω → ℝ)
    (m ε : ℝ)
    (h_meas : ∀ i, Measurable (X i))
    (h_indep : iIndepFun X P)
    (h_bound : ∀ i, ∀ᵐ ω ∂P, X i ω ∈ Set.Icc (0 : ℝ) 1)
    (hm : ∀ i, ∫ ω, X i ω ∂P = m)
    (hε : 0 < ε) :
    P.real {ω |
      m + ε ≤ (n : ℝ)⁻¹ * ∑ i, X i ω}
        ≤ Real.exp (-2 * (n : ℝ) * ε ^ 2) ∧
    P.real {ω |
      (n : ℝ)⁻¹ * ∑ i, X i ω ≤ m - ε}
        ≤ Real.exp (-2 * (n : ℝ) * ε ^ 2) := by
  have hn0 : (0 : ℝ) < n := by exact_mod_cast hn
  have hconst : ((‖(1 : ℝ) - 0‖₊ / 2) ^ 2 : NNReal) = 1 / 4 := by norm_num
  have hsub : ∀ i, HasSubgaussianMGF (fun ω => X i ω - m) (1 / 4 : NNReal) P := by
    intro i
    have h := hasSubgaussianMGF_of_mem_Icc (μ := P) (X := X i)
      (h_meas i).aemeasurable (h_bound i)
    rw [hconst, hm i] at h
    exact h
  have hsub' : ∀ i, HasSubgaussianMGF (fun ω => m - X i ω) (1 / 4 : NNReal) P := by
    intro i
    exact (hsub i).neg.congr (ae_of_all _ fun ω => by simp [neg_sub])
  have hind : iIndepFun (fun i ω => X i ω - m) P :=
    h_indep.comp (fun _ x => x - m) (fun _ => measurable_id.sub_const m)
  have hind' : iIndepFun (fun i ω => m - X i ω) P :=
    h_indep.comp (fun _ x => m - x) (fun _ => measurable_const.sub measurable_id)
  have hexp : Real.exp (-((n : ℝ) * ε) ^ 2 /
      (2 * ((∑ _i : Fin n, (1 / 4 : NNReal) : NNReal) : ℝ)))
      = Real.exp (-2 * (n : ℝ) * ε ^ 2) := by
    congr 1
    have hsum : ((∑ _i : Fin n, (1 / 4 : NNReal) : NNReal) : ℝ) = (n : ℝ) / 4 := by
      simp
      ring
    rw [hsum]
    field_simp
    ring
  have hnε : (0 : ℝ) ≤ (n : ℝ) * ε := by positivity
  constructor
  · have h := HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun hind
      (c := fun _ => (1 / 4 : NNReal)) (s := Finset.univ)
      (fun i _ => hsub i) hnε
    rw [hexp] at h
    refine le_trans (le_of_eq ?_) h
    congr 1
    ext ω
    simp only [Set.mem_setOf_eq, Finset.sum_sub_distrib, Finset.sum_const,
      Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    rw [inv_mul_eq_div, le_div_iff₀ hn0]
    constructor <;> intro h <;> nlinarith
  · have h := HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun hind'
      (c := fun _ => (1 / 4 : NNReal)) (s := Finset.univ)
      (fun i _ => hsub' i) hnε
    rw [hexp] at h
    refine le_trans (le_of_eq ?_) h
    congr 1
    ext ω
    simp only [Set.mem_setOf_eq, Finset.sum_sub_distrib, Finset.sum_const,
      Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    rw [inv_mul_eq_div, div_le_iff₀ hn0]
    constructor <;> intro h <;> nlinarith`;
const escapeHTML = value => value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
// Lightweight local Lean highlighting; works when opening index.html directly.
function highlightLean(source) {
  const tokens = /(--[^\n]*|"(?:\\.|[^"\\])*"|\b\d+(?:\.\d+)?\b|[A-Za-z_][A-Za-z_0-9]*|[ℝℕℤℚℂΩ]|[∀∃∈≤≥¬∧∨→↔λ:=+*⁻¹^])/g;
  const keywords = new Set(['theorem', 'def', 'abbrev', 'noncomputable', 'by', 'sorry', 'have', 'let', 'fun', 'intro', 'exact', 'apply', 'show', 'from', 'where']);
  const types = new Set(['Type', 'Prop', 'MeasurableSpace', 'Measure', 'IsProbabilityMeasure', 'Fin', 'Measurable', 'Set', 'Real', 'Vec', 'ℝ', 'ℕ', 'ℤ', 'ℚ', 'ℂ', 'Ω']);
  const functions = new Set(['norm_num', 'simp', 'rw', 'ring', 'iIndepFun', 'example_theorem', 'disprove_step3', 'dot', 'normSq', 'P', 'exp', 'Icc']);
  let result = '', cursor = 0;
  for (const match of source.matchAll(tokens)) {
    const token = match[0];
    result += escapeHTML(source.slice(cursor, match.index));
    const kind = token.startsWith('--') ? 'comment' : token.startsWith('"') ? 'string' : /^\d/.test(token) ? 'number' : keywords.has(token) ? 'keyword' : types.has(token) ? 'type' : functions.has(token) ? 'function' : /^[∀∃∈≤≥¬∧∨→↔λ:=+*⁻¹^]$/.test(token) ? 'operator' : 'variable';
    result += '<span class="lean-' + kind + '">' + escapeHTML(token) + '</span>';
    cursor = match.index + token.length;
  }
  return result + escapeHTML(source.slice(cursor));
}
const examples = {
  auto: {
    label: 'Problem statement · Hoeffding’s inequality',
    question: hoeffding + '<p><strong>Task:</strong> Restore the Lean statement while preserving the intended mathematical meaning. The proof is fixed to <code>sorry</code>.</p>',
    nodes: [['Read context', 'Natural language'], ['Recover assumptions', 'Explicit + implicit'], ['Write statement', 'Lean 4'], ['Check faithfulness', 'Elaboration + BEq']],
    steps: [
      ['Natural Language Statement', 'Given the NL source theorem and its context, the model must restore the text between the theorem’s name and <code>:=</code>.'],
      ['Stated in NL', 'P : Measure(Ω), P(Ω) = 1 · iIndepFun(X, P) · P(0 ≤ Xᵢ ≤ 1) = 1 · 𝔼<sub>P</sub>[Xᵢ] = μ · ε &gt; 0'],
      ['Implicit Assumptions', '(Ω, 𝓕) measurable space · Xᵢ : Ω → ℝ measurable · n ∈ ℕ<sub>&gt;0</sub>. ML theorems leave assumptions such as integrability or measurability implicit.'],
      ['From Context', 'X̄ₙ = (1/n) Σᵢ Xᵢ. We evaluate the generated statement against its gold Lean statement with BEq.']
    ],
    output: '<pre>' + escapeHTML(statement) + '</pre>'
  },
  prove: {
    label: 'Problem statement · Prove the fixed Lean theorem',
    question: hoeffding + '<p><strong>Task:</strong> Supply a complete Lean proof without changing the statement.</p>',
    nodes: [['Plan the proof', 'Use source reasoning'], ['Build upper tail', 'Markov + Hoeffding'], ['Build lower tail', 'Apply to −Yᵢ'], ['Check and repair', 'Lean feedback loop']],
    steps: [
      ['Theorem-proving', 'Given a human-verified target statement with its proof replaced by <code>sorry</code>, the model must supply a complete Lean proof.'],
      ['NL proof · steps 1–2', '1. Let Yᵢ = Xᵢ − μ. 2. Apply exponential Markov inequality.'],
      ['NL proof · steps 3–5', '3. Use independence and Hoeffding’s lemma. 4. Choose λ = 4ε. 5. Repeat for −Yᵢ (lower tail).'],
      ['Lean Verification', 'A proof counts as successful only when Lean accepts the completed proof without <code>sorry</code> or newly introduced axioms.']
    ],
    output: '<pre>' + escapeHTML(theoremProof) + '</pre>'
  },
  verify: {
    label: 'Problem statement · Find the first substantive error',
    question: String.raw`<p>For \(x,z\in\mathbb{R}^2\) with \(\langle x,(zz^{\mathsf T}-xx^{\mathsf T})x\rangle=0\), show \(\langle x,z\rangle^2=\lVert x\rVert_2^4\).</p><p><b>1.</b> \(\langle x,(zz^{\mathsf T}-xx^{\mathsf T})x\rangle=0\).</p><p><b>2.</b> \(\langle x,zz^{\mathsf T}x\rangle=(z^{\mathsf T}x)^2=\langle x,z\rangle^2\).</p><p><b>3.</b> “By Cauchy–Schwarz, \(\langle x,z\rangle^2=\lVert x\rVert^2\lVert z\rVert^2\).”</p><p><strong>Task:</strong> Locate the first substantive error, if any.</p>`,
    nodes: [['Read proof steps', 'Preserve order'], ['Formalize step 3', 'Premise → equality'], ['Construct refutation', 'x = (1,0), z = (1,1)'], ['Report first error', 'Step + evidence']],
    steps: [
      ['Formalization', 'The model must formalize every step in Lean itself, mirroring real-world verification, where reviewers receive only NL proofs.'],
      ['Formal Language Prove', 'Step 1 and Step 2 are verified; Step 3 fails with an unsolved goal. FL Prove flags the first step the model fails to prove.'],
      ['Formal Language Refute', 'Counterexample: x = (1, 0), z = (1, 1). Premise: 1 = 1; Step 3: 1 = 2. FL Refute flags a step only when Lean accepts the model’s refutation of the step’s formalization.'],
      ['Lean accepted · Step 3 refuted', 'The prediction is the earliest flagged step: Step 3.']
    ],
    output: '<pre>' + escapeHTML(`abbrev Vec₂ := ℝ × ℝ
def dot (x z : Vec₂) : ℝ := x.1 * z.1 + x.2 * z.2
def normSq (x : Vec₂) : ℝ := dot x x
def P (x z : Vec₂) : Prop :=
  (dot x z) ^ 2 = (normSq x) ^ 2 →
  (dot x z) ^ 2 = normSq x * normSq z

-- Counterexample: x = (1, 0), z = (1, 1)
-- Premise: 1 = 1; Step 3: 1 = 2
theorem disprove_step3 : ¬ P (1, 0) (1, 1) := by
  norm_num [P, dot, normSq]`) + '</pre>'
  }
};
// Playback: each step streams into the walkthrough box, then the final output streams into the terminal.
// The example then loops. The parent page picks the example via postMessage.
const STEP_STREAM_MS = 2000;  // time to stream one step
const STEP_HOLD_MS = 2000;    // pause after a step is fully shown
const FADE_MS = 600;          // fade-out before the next step
const LOOP_PAUSE_MS = 4000;
const $ = id => document.getElementById(id);
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
let current = 'auto', run = 0, timer = null, frame = null;
const FRAME_MS = 30;

function clearPlayback() {
  run++;
  clearTimeout(timer); timer = null;
  if (frame !== null) { clearTimeout(frame); frame = null; }
}

// Reveal `markup` into `target` character by character while keeping its tags.
function streamInto(target, markup, duration, done, scroller) {
  const template = document.createElement('template');
  template.innerHTML = markup;
  const walker = document.createTreeWalker(template.content, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) if (!walker.currentNode.parentElement || !walker.currentNode.parentElement.closest('.n')) nodes.push({ node: walker.currentNode, text: Array.from(walker.currentNode.textContent) });
  nodes.forEach(n => { n.node.textContent = ''; });
  target.append(template.content);
  const total = nodes.reduce((sum, n) => sum + n.text.length, 0);
  const finish = () => { nodes.forEach(n => { n.node.textContent = n.text.join(''); }); if (scroller) scroller.scrollTop = scroller.scrollHeight; done(); };
  if (reducedMotion || total === 0) { finish(); return; }
  const id = run;
  const start = performance.now();
  let shown = 0;
  // Timers rather than requestAnimationFrame, so pacing holds even when the frame is not painted.
  function tick() {
    if (id !== run) return;
    const now = performance.now();
    const count = Math.min(total, Math.floor((now - start) / duration * total));
    if (count > shown) {
      const follow = !scroller || scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 40;
      let left = count;
      for (const n of nodes) {
        const take = Math.min(n.text.length, left);
        if (n.node.textContent.length !== take) n.node.textContent = n.text.slice(0, take).join('');
        left -= take;
        if (left <= 0) break;
      }
      shown = count;
      if (scroller && follow) scroller.scrollTop = scroller.scrollHeight;
    }
    if (shown < total) frame = setTimeout(tick, FRAME_MS);
    else { frame = null; done(); }
  }
  frame = setTimeout(tick, FRAME_MS);
}

function later(fn, ms) {
  const id = run;
  timer = setTimeout(() => { if (id === run) fn(); }, ms);
}

function playStep(index) {
  const item = examples[current];
  if (index === item.steps.length) { playOutput(); return; }
  const [title, body] = item.steps[index];
  // One step at a time: stream it, hold, fade it out, then show the next one.
  const entry = document.createElement('div');
  entry.className = 'step';
  $('trace').replaceChildren(entry);
  $('trace').scrollTop = 0;
  const last = index === item.steps.length - 1;
  streamInto(entry, '<h3><span class="n">' + (index + 1) + '</span>' + title + '</h3><p>' + body + '</p>', STEP_STREAM_MS, () => {
    later(() => {
      if (last) { playStep(index + 1); return; }  // keep the last step visible while the output streams
      entry.classList.add('leaving');
      later(() => playStep(index + 1), FADE_MS);
    }, STEP_HOLD_MS);
  }, $('trace'));
}

function playOutput() {
  const item = examples[current];
  const output = $('output');
  output.className = 'output-body fade';
  output.replaceChildren();
  output.scrollTop = 0;
  const template = document.createElement('template');
  template.innerHTML = item.output;
  const source = template.content.querySelector('pre').textContent;
  const code = document.createElement('pre');
  output.append(code);
  $('output-status').textContent = 'Streaming…';
  const duration = Math.min(16000, Math.max(2500, source.length * 5));
  streamInto(code, escapeHTML(source), duration, () => {
    $('output-status').textContent = 'Ready';
    later(() => selectTask(current), LOOP_PAUSE_MS);
  }, output);
}

// The walkthrough box does not scroll: size it to the tallest step of the current example.
function fitTrace() {
  const trace = $('trace');
  const probe = trace.cloneNode(false);
  probe.removeAttribute('id');
  probe.style.cssText = 'position:absolute;visibility:hidden;height:auto;left:-9999px;width:' + trace.clientWidth + 'px';
  document.body.append(probe);
  let tallest = 0;
  examples[current].steps.forEach(([title, body], i) => {
    probe.innerHTML = '<div class="step"><h3><span class="n">' + (i + 1) + '</span>' + title + '</h3><p>' + body + '</p></div>';
    tallest = Math.max(tallest, probe.offsetHeight);
  });
  probe.remove();
  trace.style.height = tallest + 'px';
}
let resizeTimer = null;
addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(fitTrace, 100); });
function selectTask(key) {
  clearPlayback();
  current = key;
  document.body.dataset.task = key;  // proof verification uses the green palette
  const item = examples[key];
  $('question').innerHTML = item.question;
  renderMathInElement($('question'), {throwOnError: false});
  fitTrace();
  $('trace').replaceChildren();
  $('output').className = 'output-body waiting';
  $('output').textContent = 'The final output appears after the last step.';
  $('output-status').textContent = 'Pending';
  playStep(0);
}

window.addEventListener('message', event => {
  const data = event.data;
  if (data && data.type === 'mlp-select-task' && examples[data.task]) selectTask(data.task);
});
selectTask(examples[location.hash.slice(1)] ? location.hash.slice(1) : current);
