# Directional Damage in Continual Learning — Complete Project Specification

**A research project on whether "future-trainability damage" has an independently controllable directional component**

Version 1.0 · Solo researcher · Single consumer GPU (RTX 4070, 8 GB)

---

## How to read this document

This file is written in two registers on purpose:

- **Plain-English boxes** (marked 🟢 **In simple terms**) explain each idea as if to a smart friend who knows some machine learning but nothing about this subfield.
- **Technical sections** give the precise definitions, math, controls, and code plan you actually need to run the project.

If you only read the plain-English boxes top to bottom, you will understand *what* the project is and *why* it matters. If you read everything, you will be able to *build* it.

---

## Table of Contents

1. [The one-paragraph summary](#1-the-one-paragraph-summary)
2. [The problem, explained simply](#2-the-problem-explained-simply)
3. [Background: the three things everyone already knows](#3-background)
4. [The gap: what nobody has isolated yet](#4-the-gap)
5. [The core hypothesis](#5-the-core-hypothesis)
6. [Key definitions (the descriptors)](#6-key-definitions)
7. [Why this is hard — and why that makes it worth doing](#7-why-this-is-hard)
8. [The experimental design overview](#8-experimental-design-overview)
9. [Candidate B — the decisive discovery experiment](#9-candidate-b)
10. [Candidate A — the method-building phase](#10-candidate-a)
11. [The control experiments (this is where the project lives or dies)](#11-controls)
12. [The statistics: the β₄ regression and multicollinearity](#12-statistics)
13. [The six kill conditions](#13-kill-conditions)
14. [Datasets — what they are, why, and where to download them](#14-datasets)
15. [Hardware feasibility](#15-hardware)
16. [Week-by-week plan](#16-plan)
17. [Baselines and prior work to compare against](#17-baselines)
18. [Metrics](#18-metrics)
19. [What success and failure each look like](#19-success-failure)
20. [Honest self-assessment and scores](#20-scores)
21. [Paper structure](#21-paper-structure)
22. [Glossary](#22-glossary)

---

<a name="1-the-one-paragraph-summary"></a>
## 1. The one-paragraph summary

When a neural network learns a sequence of tasks, learning each new task damages its ability to learn *future* tasks — this is called **loss of plasticity**. The dominant explanation says the damage is essentially about *how far* the weights moved: big or long updates use up the network's flexibility. This project asks a sharper question: **on top of how far the weights moved, does the *direction* they moved in independently determine how much future-learning ability is lost?** If we hold the size of the update fixed, hold the benefit to the current task fixed, and hold the damage to past tasks fixed — is there *still* a leftover, controllable "directional" effect on future trainability? If yes, that is a new knob nobody has documented, and it points toward better continual-learning methods. If no, we have a clean negative result that sharpens the field's understanding.

> 🟢 **In simple terms**
> Imagine your brain's capacity to learn is like a fresh field of snow. Every time you learn something, you leave footprints. The standard belief is: *the more footprints you stamp (and the harder you stamp), the harder it becomes to make clean new prints later.* We're asking a different question: *does the direction you walk matter too?* Maybe walking north ruins the field for future walkers, but walking east — the same number of steps, same force — leaves it much more usable. If direction matters independently of the number of steps, that's a discovery. If it doesn't, we've proven the footprint-count theory was right all along. Both answers are useful.

---

<a name="2-the-problem-explained-simply"></a>
## 2. The problem, explained simply

Neural networks are trained by nudging their internal numbers (**weights**) to reduce error. Do this on Task 1, then Task 2, then Task 3, and two bad things happen:

1. **Catastrophic forgetting** — the network gets worse at Task 1 after learning Task 2. (This is about the *past*.)
2. **Loss of plasticity** — the network gets *slower and worse at learning new tasks* the more tasks it has already seen. It becomes rigid. (This is about the *future*.)

This project is about #2, loss of plasticity. It is the less-understood of the two and arguably the more fundamental, because a network that can't learn anything new is useless regardless of what it remembers.

> 🟢 **In simple terms**
> Two different problems get confused constantly. "Forgetting" is when learning French makes you worse at the Spanish you already knew. "Loss of plasticity" is when, after learning ten languages, your brain has somehow become bad at learning an *eleventh* language you've never seen — even though the new language has nothing to do with the old ones. We're studying the second one: the mysterious way that learning *uses up your ability to keep learning*.

Here's the subtle part. When people study loss of plasticity, they usually explain it by pointing at **the magnitude of the changes**: the weights drifted far from where they started, or certain neurons "died" (stopped responding), or the useful directions in the network's representation collapsed. All of these are, roughly, *size* or *amount* stories. Very little work asks whether the **geometric direction** of the weight update — which way in the high-dimensional weight space you moved — carries its own independent effect once you've accounted for size.

That's the wedge this project drives into.

---

<a name="3-background"></a>
## 3. Background: the three things everyone already knows

To see the gap, you need to know the three well-established results this project sits next to. Each of these is real, published, and *not* what we're claiming — we're claiming something they leave open.

### 3.1 Catastrophic overtraining / progressive sensitivity (magnitude & duration)

Recent work (e.g., the Springer et al. 2025 line on catastrophic overtraining) establishes that the *more* and the *longer* you train, the more fragile the network becomes to subsequent updates. The damage is attributed to **magnitude and duration** of training — not to direction.

> 🟢 **In simple terms**
> "Train too much and the model becomes brittle." True and important — but it's a story about *how much*, not *which way*.

### 3.2 Directions of curvature (mechanism, but retrospective and scalar)

Work like Lewandowski et al. on the directions of curvature in the loss landscape gets closer to geometry: it looks at *where* the landscape is curved and connects that to plasticity. But this analysis is largely **retrospective** (it explains loss of plasticity after the fact) and **scalar** (it summarizes curvature into single numbers rather than isolating a controllable directional cause you can intervene on).

> 🟢 **In simple terms**
> Some researchers have looked at the "shape" of the learning landscape and linked its curvature to rigidity. But they mostly *describe* what happened, using summary numbers. They don't grab a specific directional knob and *turn it* to prove it causes the damage.

### 3.3 Null-space / safe-direction methods (protect the past, not the future)

A whole family of methods (null-space projection, orthogonal-gradient methods, etc.) constrains updates to directions that don't disturb *previously learned* tasks. These are directional — but their entire aim is protecting the **past** (reducing forgetting). None of them target the **future** (preserving the ability to learn tasks not yet seen).

> 🟢 **In simple terms**
> There are clever methods that say "only change the network in ways that won't wreck what it already knows." That protects old memories. Nobody has a method that says "only change the network in ways that won't wreck its ability to learn things it *hasn't seen yet*." That future-facing direction is unclaimed.

**The pattern:** magnitude stories dominate the *future-trainability* question, and the directional stories that exist are either retrospective, scalar, or aimed at the past. So the specific object — **an independently controllable directional component of *future*-trainability damage** — has not been isolated.

---

<a name="4-the-gap"></a>
## 4. The gap: what nobody has isolated yet

Put precisely, the research gap is:

> **Does future-trainability damage contain a directional component that remains controllable *after* simultaneously controlling for (a) update magnitude, (b) current-task benefit, and (c) past-task damage?**

Every neighboring result controls for *some* of these but not all three at once, and none isolates the *future*-facing directional residual as a thing you can intervene on. That simultaneous control is the whole contribution. It's also the whole difficulty (see §7 and §12).

> 🟢 **In simple terms**
> Lots of people have studied pieces of this. But nobody has run the clean experiment where you *pin down everything else* — same step size, same benefit to the current task, same damage to old tasks — and then check whether direction *still* independently predicts how much future learning ability you lost. If you don't pin down all three at once, you can't tell whether "direction" is doing anything or is just a stand-in for one of the others.

---

<a name="5-the-core-hypothesis"></a>
## 5. The core hypothesis

Stated in the required testable form:

> **We hypothesize that constraining the *directional* component of parameter updates will reduce future-trainability damage — at matched update magnitude and matched past-task damage — because interference has a directional term not captured by magnitude-based accounts.**

Two properties make this a good hypothesis:

- It is **falsifiable**: if the directional residual is statistically zero under matched controls, the hypothesis is dead, and we say so.
- Its outcome is **not obvious**: strong, credible prior work (§3.1) predicts magnitude explains everything, which would make our residual vanish. We are genuinely uncertain, which is the point.

> 🟢 **In simple terms**
> Our bet: "direction matters on its own." The honest counter-bet from existing research: "no, it's all just about size — control for size and direction stops mattering." We don't know which is true. That uncertainty is exactly why it's worth an experiment instead of an opinion.

---

<a name="6-key-definitions"></a>
## 6. Key definitions (the descriptors)

These must be **locked before looking at any outcomes** (pre-registration; see §16). Vague definitions here are how projects like this fool themselves.

| Symbol | Name | Plain meaning | Operational definition (to be finalized in pre-registration) |
|---|---|---|---|
| **Δθ** | Update vector | "The change made to the weights" while learning a task | θ_after − θ_before over a defined training segment |
| **‖Δθ‖** | Update magnitude | "How big the change was" | L2 norm of Δθ (this is what we hold fixed) |
| **d = Δθ / ‖Δθ‖** | Update direction | "Which way we moved," size stripped out | Unit vector of the update |
| **FTD** | Future-Trainability Damage | "How much harder a fresh future task became to learn" | Drop in learning speed / attainable performance on a held-out probe task, relative to a reference network |
| **CTB** | Current-Task Benefit | "How much the current task improved" | Loss reduction on the current task over the segment |
| **PTD** | Past-Task Damage | "How much old tasks got worse" (forgetting) | Increase in loss on previously learned tasks |
| **directional residual** | The thing we're hunting | "The part of FTD explained by direction *after* removing size, current benefit, and past damage" | The coefficient/effect attributable to **d** once ‖Δθ‖, CTB, PTD are controlled |

> 🟢 **In simple terms**
> - **Δθ**: the edit you made to the network.
> - **‖Δθ‖**: how big that edit was.
> - **d**: purely the *direction* of the edit, ignoring how big it was.
> - **FTD**: how much you hurt the network's ability to learn something new next.
> - **CTB**: how much the current lesson helped.
> - **PTD**: how much you damaged old memories.
> - **directional residual**: the leftover influence of *direction* on future-learning-damage once we've subtracted out size, current benefit, and old-memory damage. This leftover is the entire prize.

### How we measure "future-trainability damage" (FTD) concretely

1. Take the network at a checkpoint.
2. Apply an update (learning the current task) to get a new network.
3. Give **both** networks a brand-new **probe task** they've never seen.
4. Let each learn the probe for a short, fixed budget.
5. FTD = how much *worse/slower* the updated network learns the probe compared to the reference.

The probe task is the instrument. Its design (how novel, how many, fixed seeds) is load-bearing and gets pre-registered.

---

<a name="7-why-this-is-hard"></a>
## 7. Why this is hard — and why that makes it worth doing

The difficulty is **confounding**. The four quantities — magnitude, direction, current benefit, past damage — naturally move together. If you just train normally and correlate direction with FTD, you can't tell whether "direction" is real or is secretly encoding "the update was bigger" or "the update helped the current task more."

So the project is really an exercise in **controlled intervention and clean statistics**, not in building a fancy architecture. The novelty is *epistemic*: constructing a setting where the directional residual can be isolated and either confirmed or ruled out.

> 🟢 **In simple terms**
> The hard part isn't inventing a cool network. It's designing an experiment careful enough that, when direction *looks* important, we can be sure it isn't just size or benefit in disguise. This is detective work: eliminate every other suspect, then see if the remaining suspect (direction) actually did anything.

This is also why the two most fragile sections — **past-task damage matching (Control C)** and **multicollinearity in the β₄ regression** — get their own sections below. They are the two places the whole argument can quietly break.

---

<a name="8-experimental-design-overview"></a>
## 8. The experimental design overview

The project is **gated into two experiments**. You do not proceed to the second unless the first passes.

```
                ┌─────────────────────────────────────────┐
                │  CANDIDATE B  (discovery / decisive)      │
                │  "Does a directional residual even exist  │
                │   under matched controls?"                │
                └───────────────────┬─────────────────────┘
                                    │
                 pass (residual is real & robust)
                                    │
                                    ▼
                ┌─────────────────────────────────────────┐
                │  CANDIDATE A  (method-building)           │
                │  "Can we exploit the directional knob to  │
                │   build a method that reduces plasticity  │
                │   loss at matched magnitude?"             │
                └─────────────────────────────────────────┘

     fail Candidate B  →  STOP. Report the clean negative result.
                          (This is still publishable and honest.)
```

> 🟢 **In simple terms**
> Step one: *prove the knob exists.* Step two (only if step one works): *build something useful with the knob.* If step one fails, we don't fake a method — we honestly report "the knob isn't there," which itself corrects the field.

---

<a name="9-candidate-b"></a>
## 9. Candidate B — the decisive discovery experiment

**Question:** After matching magnitude, current-task benefit, and past-task damage, is there a nonzero, robust relationship between update *direction* and future-trainability damage?

**Shape of the experiment:**

1. **Generate many updates** from the same starting checkpoint that are *matched* on ‖Δθ‖ (same size), and where you can also measure or match CTB and PTD.
2. For each update, measure **FTD** via the probe protocol (§6).
3. Regress FTD on a directional descriptor while controlling for ‖Δθ‖, CTB, PTD (the **β₄ regression**, §12).
4. Test whether the directional coefficient (β₄) is significantly nonzero and survives robustness checks.

**How to get many matched-magnitude, different-direction updates (options to pre-register one):**

- Perturb the optimizer's trajectory with different random seeds / data orderings, then rescale updates to a common norm.
- Inject controlled directional perturbations orthogonal to a reference update and renormalize.
- Sample updates under different regularizers that change direction while you post-hoc match norm.

The exact generator is a design choice locked in pre-registration; what matters is that **magnitude is held fixed by construction** and direction varies.

> 🟢 **In simple terms**
> Make a big batch of edits that are all the *same size* but point in *different directions*. For each one, check how much it hurt future learning. Then statistically ask: once size, current-benefit, and old-damage are accounted for, does direction still explain any of the future-learning damage? That's the whole discovery experiment.

**Why synthetic data first (see §14):** on a synthetic teacher-student setup you can control task structure exactly, generate unlimited matched samples cheaply, and know the ground truth — which is essential for trusting the statistics before touching messier real data.

---

<a name="10-candidate-a"></a>
## 10. Candidate A — the method-building phase (only if B passes)

**Question:** If the directional knob is real, can we design an update rule that steers *away* from the harmful directional component — at matched magnitude — and thereby preserves plasticity better than magnitude-only controls?

**Shape:**

- Define a directional constraint or penalty that discourages movement along the empirically-identified harmful direction.
- Train continual sequences with this constraint vs. baselines matched on update magnitude.
- Measure whether plasticity (future-task learning speed/accuracy) improves *without* simply shrinking the updates (which would be the trivial, already-known effect).

**Critical fairness control:** the method must beat baselines **at matched ‖Δθ‖**. If it only "works" by taking smaller steps, that's the magnitude story again and not a contribution.

> 🟢 **In simple terms**
> Once we know a harmful direction exists, build a training rule that gently avoids it — while taking steps of the *same size* as normal. If the network then stays better at learning new things, we've turned a discovery into a tool. The catch: it has to win *without cheating by taking tinier steps*, because "take smaller steps" is old news.

---

<a name="11-controls"></a>
## 11. The control experiments (this is where the project lives or dies)

Three controls correspond to the three things we must hold fixed. Two are routine; **Control C is the dangerous one.**

- **Control A — Magnitude matching.** Hold ‖Δθ‖ fixed across compared updates. *Routine*: enforced by renormalization.
- **Control B — Current-task benefit.** Account for CTB so a direction isn't credited/blamed for simply helping the current task more. *Manageable*: measure and include as a covariate, and/or bin by CTB.
- **Control C — Past-task damage matching.** Ensure compared updates inflict the *same* PTD, so the directional residual isn't just forgetting wearing a disguise. **Fragile**: matching PTD tightly across many updates is genuinely hard, and loose matching silently confounds the result.

**Why Control C is load-bearing:** if updates that point in the "harmful" future direction also happen to cause more past-task damage, and you *don't* match PTD tightly, you'll credit "direction → future damage" when the truth might be "direction → past damage, which we failed to hold fixed." The directional residual would be an artifact.

**Mitigations to pre-register:**
- Tight PTD binning / caliper matching; discard unmatched updates.
- Report results across several PTD-matching tolerances to show the residual isn't an artifact of one tolerance.
- Include PTD as a regression covariate *and* match on it (belt-and-suspenders).

> 🟢 **In simple terms**
> To trust "direction hurts future learning," we must be sure the compared edits did the *same* amount of damage to old memories. If "bad-direction" edits also secretly trash old memories more, we might mistake old-memory damage for future-learning damage. Forcing equal old-memory damage across edits is the hardest part of the whole project — and if we can't do it cleanly, the finding isn't safe.

---

<a name="12-statistics"></a>
## 12. The statistics: the β₄ regression and multicollinearity

The central test is a regression of the form (schematically):

```
FTD  =  β0
      + β1 · ‖Δθ‖        (magnitude)
      + β2 · CTB          (current-task benefit)
      + β3 · PTD          (past-task damage)
      + β4 · D_dir        (directional descriptor)   ← the coefficient we care about
      + error
```

The claim "there is an independently controllable directional component" is essentially the claim **β₄ ≠ 0 and robust.**

**The threat: multicollinearity.** If D_dir is strongly correlated with ‖Δθ‖, CTB, or PTD across your samples, the regression *cannot* cleanly attribute variance to β₄. The coefficient becomes unstable (huge standard errors, sign flips across subsamples), and the central claim becomes **statistically unidentifiable** — not false, but unprovable with this design. This is an existential risk, not a cosmetic one.

**Diagnostics and mitigations (pre-register):**
- Compute **Variance Inflation Factors (VIF)** for all covariates; set a threshold above which the design is declared under-identified.
- **Construct** the update generator to *decorrelate* D_dir from the others (e.g., sample directions orthogonal to the magnitude/benefit gradients by design).
- Run an **identifiability pre-check on synthetic data first**: before any intervention coding, confirm the four terms are separable enough to regress. If VIF is hopeless, that's the earliest, cheapest kill signal — better to hit it in week 1 than month 3.
- Report **stability under subsampling / bootstrap** for β₄, not just a single p-value.

> 🟢 **In simple terms**
> We measure four things and ask a statistical model, "how much does *each* independently affect future-learning damage?" The danger: if "direction" is too tangled up with "size" or "benefit" or "old-damage" in our data, the math literally can't separate them — like trying to tell how much two singers each contributed when they only ever sang in perfect unison. The fix is to deliberately design the experiment so direction varies *independently* of the others, and to check (cheaply, first, on synthetic data) that the separation is even possible before investing months.

---

<a name="13-kill-conditions"></a>
## 13. The six kill conditions (falsification gates)

The project **ends or pivots** if any of these hold. Committing to these in advance is what keeps the project honest and prevents rationalization drift.

1. **Identifiability failure.** VIF/collinearity makes β₄ unestimable even after decorrelation-by-design → the design cannot answer the question; stop or redesign the generator.
2. **Null directional residual.** β₄ is statistically indistinguishable from zero under matched controls, robustly across tolerances → the magnitude account wins; report the clean negative.
3. **Control C failure.** PTD cannot be matched tightly enough to rule out forgetting-confound → the residual is not trustworthy; stop.
4. **Non-robustness.** β₄ exists but flips sign / vanishes under reasonable variations (probe choice, generator, seed) → not a real effect.
5. **Synthetic-only artifact.** A residual appears on synthetic data but completely fails to appear on Split-MNIST/CIFAR under the same protocol → likely an artifact of the synthetic construction, not a general phenomenon.
6. **Trivial-magnitude explanation for Candidate A.** Any plasticity improvement in the method phase disappears once you match ‖Δθ‖ → the method is just "smaller steps," which is already known; no contribution.

> 🟢 **In simple terms**
> We wrote down six specific ways the project is allowed to "fail," and we agreed *in advance* to accept those failures instead of explaining them away. Examples: the math can't separate direction from size (kill 1); direction turns out not to matter (kill 2); we can't equalize old-memory damage (kill 3); the effect is a fluke that changes when we poke it (kill 4); it only shows up on toy data (kill 5); or our "method" only works by taking smaller steps (kill 6). Pre-committing to these is how a scientist avoids fooling themselves.

---

<a name="14-datasets"></a>
## 14. Datasets — what they are, why, and where to download them

This project uses **one synthetic dataset you generate yourself** (primary) plus **two standard public datasets** (secondary/optional) turned into continual-learning benchmarks via the "Split" protocol.

### 14.1 Synthetic teacher–student (PRIMARY — you generate this, no download)

**What it is:** You create a fixed random "teacher" neural network. It maps random input vectors to outputs. The "student" network must learn to imitate the teacher. To make a *sequence* of tasks, you change the teacher (or the input distribution) in controlled ways between tasks.

**Why it's primary:**
- **Total control** over task structure, similarity, and difficulty.
- **Unlimited matched samples** — essential for the β₄ regression and for decorrelating direction from magnitude.
- **Known ground truth** — you know exactly how tasks relate, so you can trust the statistics before touching messy data.
- **Tiny and fast** — runs trivially on an RTX 4070.

**How to build it:** no download needed. In PyTorch: instantiate a fixed-seed MLP as the teacher; sample inputs from a chosen distribution (e.g., standard Gaussian); generate (input, teacher-output) pairs on the fly; define task changes as controlled transformations of the teacher's weights or the input covariance. (You will specify exact dimensions/seeds in the pre-registration.)

> 🟢 **In simple terms**
> You build a "correct answer machine" (the teacher) with random settings, and train a student to copy it. Because you built the teacher, you know everything about the task and can make as many practice examples as you want — perfect for careful statistics. Nothing to download; it's a few lines of PyTorch.

### 14.2 Split-MNIST (SECONDARY — real data, standard benchmark)

**What it is:** MNIST is 70,000 grayscale 28×28 images of handwritten digits 0–9 (60,000 train / 10,000 test). The **Split-MNIST** protocol turns it into a continual-learning benchmark by splitting the ten digits into **five sequential two-class tasks**: {0,1}, {2,3}, {4,5}, {6,7}, {8,9}. You learn them one after another and measure forgetting and plasticity. The split is standard in the continual-learning literature.

**Why use it:** it's the field's canonical sanity check. If your directional residual is real, it should show up here too (and if it *only* shows up on synthetic data, that's kill condition #5).

**Where to get the raw MNIST data:**

- **Via PyTorch (recommended, one line):** `torchvision.datasets.MNIST(root, download=True)` — the loader fetches and caches it for you. Docs: <https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html>
- **Via TensorFlow/Keras:** `tf.keras.datasets.mnist.load_data()` — docs: <https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data>
- **Hugging Face Datasets:** <https://huggingface.co/datasets/ylecun/mnist>
- **OpenML MNIST (mnist_784):** <https://www.openml.org/d/554>
- **Original site (Yann LeCun's page):** <http://yann.lecun.com/exdb/mnist/> (note: this site is sometimes intermittently unavailable; the loaders above are more reliable)

**Where to get the "Split" version:** Split-MNIST is not a separate download — it is a *protocol applied to MNIST*. You implement it by filtering MNIST by label into the five class-pairs. Ready-made continual-learning libraries implement it for you:

- **Avalanche** (continual-learning library) `SplitMNIST` benchmark — <https://avalanche.continualai.org/>
- **continuum** library `SplitMNIST` — <https://github.com/Continvvm/continuum>

> 🟢 **In simple terms**
> MNIST is the famous "handwritten digits" dataset. "Split-MNIST" just means: don't learn all ten digits at once — learn them two at a time, in five rounds, so the network experiences a *sequence* of tasks. You don't download a special file; you download normal MNIST (one line of code) and chop it into five pieces by digit. Libraries like Avalanche do the chopping for you.

### 14.3 Split-CIFAR (OPTIONAL — harder real data)

**What it is:** CIFAR-10 is 60,000 color 32×32 images in 10 classes (50,000 train / 10,000 test); CIFAR-100 has 100 classes. **Split-CIFAR** applies the same "learn a few classes at a time" protocol, and is meaningfully harder than Split-MNIST — a good stress test if earlier stages succeed.

**Where to get the raw CIFAR data:**

- **Via PyTorch:** `torchvision.datasets.CIFAR10(root, download=True)` / `CIFAR100(...)` — docs: <https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html>
- **Original site (Krizhevsky, CIFAR-10/100):** <https://www.cs.toronto.edu/~kriz/cifar.html>
- **Hugging Face:** <https://huggingface.co/datasets/uoft-cs/cifar10> and <https://huggingface.co/datasets/uoft-cs/cifar100>

**Split version:** same as MNIST — a protocol, implemented via Avalanche (`SplitCIFAR10` / `SplitCIFAR100`) or continuum.

> 🟢 **In simple terms**
> CIFAR is like MNIST's harder cousin — small color photos (cats, planes, etc.) instead of digits. "Split-CIFAR" chops it into sequential tasks the same way. Only reach for this once the easier datasets have already shown something, because it's more demanding.

### 14.4 Dataset summary table

| Dataset | Role | Download needed? | Primary link |
|---|---|---|---|
| Synthetic teacher–student | Primary (discovery + statistics) | No — you generate it | *n/a (PyTorch code)* |
| MNIST → Split-MNIST | Secondary (real-data sanity check) | Yes (auto via loader) | <https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html> · <https://huggingface.co/datasets/ylecun/mnist> · <https://www.openml.org/d/554> |
| CIFAR-10/100 → Split-CIFAR | Optional (harder stress test) | Yes (auto via loader) | <https://www.cs.toronto.edu/~kriz/cifar.html> · <https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html> |
| Continual-learning libraries (implement the "Split" protocols) | Tooling | Yes (pip) | Avalanche <https://avalanche.continualai.org/> · continuum <https://github.com/Continvvm/continuum> |

---

<a name="15-hardware"></a>
## 15. Hardware feasibility (RTX 4070, 8 GB)

- **Synthetic teacher–student:** trivial. Small MLPs, tiny memory, thousands of matched updates feasible. This is where 90% of the decisive work happens, and it fits comfortably.
- **Split-MNIST:** small MLP/CNN, 5 tasks — comfortable on 8 GB.
- **Split-CIFAR:** heavier but still feasible with modest CNNs and reasonable batch sizes; the optional stress test, not the core.

The project was deliberately scoped so the **decisive experiment (Candidate B on synthetic data) runs on a single consumer GPU**. Nothing here needs multi-GPU or cloud infrastructure. This is a feature: it keeps the work reproducible by an individual and keeps iteration fast.

> 🟢 **In simple terms**
> Your gaming-class GPU is plenty. The important experiment uses tiny networks and toy data, so it's fast and cheap. The harder color-image test is optional and still runs on one card. No supercomputer required — on purpose.

---

<a name="16-plan"></a>
## 16. Week-by-week plan

**Before anything: lock the pre-registration.** Write down, and freeze, the exact descriptors (§6), the FTD probe protocol, the update generator, the PTD-matching tolerance, the VIF threshold, and every kill condition — *before* looking at outcomes. This is the single most important discipline in the project.

- **Week 0 — Pre-registration.** Freeze definitions, thresholds, kill conditions.
- **Week 1 — Synthetic scaffold.** Build teacher–student pipeline; implement update generator with matched ‖Δθ‖; implement the FTD probe.
- **Week 1–2 — Identifiability pre-check (cheap kill test).** On synthetic data, confirm D_dir can be decorrelated from ‖Δθ‖/CTB/PTD enough to regress (VIF check). *If it can't, you've hit kill #1 in week 1 — the cheapest possible failure.*
- **Week 2–3 — Candidate B core.** Generate matched updates, measure FTD, run the β₄ regression with full controls (esp. Control C), robustness/bootstrap.
- **Decision gate.** Pass → proceed. Fail → write up the clean negative result.
- **Week 4+ — Split-MNIST replication** of the Candidate B protocol (guards against kill #5).
- **Later — Candidate A** method-building (only if B robustly passes), with matched-magnitude fairness control.
- **Optional — Split-CIFAR** stress test.

> 🟢 **In simple terms**
> First write the rules down and don't change them (so you can't fool yourself). Then build the toy setup. Then do the *cheapest* possible check — can the math even separate direction from size? If not, stop now. If yes, run the real discovery experiment. Only if that clearly works do you build a method and try it on harder data.

---

<a name="17-baselines"></a>
## 17. Baselines and prior work to compare against

For the **discovery** phase, the "baseline" is really the *competing explanation*: magnitude/duration accounts (§3.1). Your job is to show direction adds explanatory power *beyond* magnitude.

For the **method** phase (Candidate A), compare against:

- **Magnitude-only control** — the same training but with matched update norms and no directional constraint (the critical fairness baseline).
- **Reset-based plasticity methods** — e.g., Continual Backprop-style selective resets (a leading family that actually mitigates plasticity loss).
- **Curvature/direction-of-curvature diagnostics** — as a conceptual comparison point (retrospective, scalar) to position your controllable, prospective directional knob.
- **Null-space / orthogonal-gradient methods** — to contrast "protect the past" (their aim) with "protect the future" (yours).

> 🟢 **In simple terms**
> In the discovery phase, the thing you're "beating" is the popular belief that only size matters — you win by showing direction explains something extra. In the method phase, you compare against: plain training with the same step sizes (the honest control), the best existing anti-rigidity tricks (like resetting dead neurons), and the past-protecting methods (to highlight that yours protects the *future* instead).

---

<a name="18-metrics"></a>
## 18. Metrics

- **Directional residual significance:** β₄ estimate, its confidence interval, and bootstrap stability — the primary outcome.
- **Identifiability:** VIF for all covariates (a gate, not just a report).
- **Future-trainability:** probe-task learning speed and attained accuracy (defines FTD).
- **Forgetting:** standard **Backward Transfer (BWT)** and average accuracy (ACC) on Split-MNIST/CIFAR for the replication and method phases.
- **Matching quality:** distributions of ‖Δθ‖, CTB, PTD across compared groups, to *prove* the controls actually held.
- **Method effect (Candidate A):** improvement in plasticity at **matched ‖Δθ‖** vs. baselines (the only version that counts).

> 🟢 **In simple terms**
> The headline number is whether "direction" has a real, stable effect (β₄). We also constantly report evidence that our controls worked (the sizes/benefits/old-damage really were equalized), plus standard continual-learning scores (how much was forgotten, how accurate overall). For the method, the only score that matters is: did it help *without* just shrinking the steps?

---

<a name="19-success-failure"></a>
## 19. What success and failure each look like

**Clean success (best case):** β₄ is significantly nonzero, robust across probes/generators/tolerances, survives on synthetic *and* Split-MNIST, controls verifiably held, and (Candidate A) a directional constraint improves plasticity at matched magnitude. → A genuine new knob + a method. Strong paper.

**Partial success (still good):** β₄ real and robust on synthetic and Split-MNIST, but Candidate A doesn't yet yield a practical method. → A solid "here is a previously-unisolated directional component" findings paper.

**Clean negative (still publishable):** careful controls show β₄ ≈ 0 robustly. → "Future-trainability damage is (surprisingly) magnitude-dominated once you control properly" — a real correction to loose directional intuitions in the field, and an honest contribution.

**Non-result (not publishable):** identifiability failure or Control C failure means you *can't tell* → you learned the design was inadequate; you pivot the generator or the setting. Not a paper, but caught early and cheaply by design.

> 🟢 **In simple terms**
> Best case: direction matters and we can use it. Good case: direction matters but we haven't built a tool yet. Still-worth-it case: direction *doesn't* matter and we prove it cleanly, correcting a common assumption. Bad case: our experiment couldn't separate the effects at all — but we're designed to catch that in week one, so it costs us little.

---

<a name="20-scores"></a>
## 20. Honest self-assessment and scores

Scored on a 1–10 scale, deliberately critical.

| Dimension | Score | Honest reasoning |
|---|---|---|
| Novelty | 7 | The *simultaneous* control isolating a future-facing directional residual is genuinely unclaimed; neighbors control subsets. Not a 9 because the surrounding concepts (curvature, magnitude, null-space) are heavily studied — the novelty is in the clean isolation, not new machinery. |
| Technical depth | 8 | The difficulty is real and conceptual (identifiability, controlled intervention, careful statistics), not superficial. |
| Importance | 8 | Loss of plasticity is a core, unsolved obstacle to continual learning; a new controllable axis would matter. |
| Real-world applicability | 7 | If the knob is real and Candidate A works, it feeds directly into practical continual-learning methods; contingent on the discovery holding. |
| Feasibility | 8 | Fits a single consumer GPU; decisive experiment is synthetic and cheap. |
| Experimental risk | (high) | Honestly high: multicollinearity and Control C can each sink it. Mitigated by early, cheap kill checks. |
| **Paper potential** | **7 (not 8)** | Outcome-gated. It becomes an 8 only when Candidate B returns a clean, robust directional residual under matched controls. Until then, honesty demands 7. |

**What would make reviewers reject this?**
- "Your β₄ is not identifiable — the covariates are collinear." (The #1 risk. Pre-empt with VIF gates and decorrelation-by-design.)
- "Your Control C is too loose — this is just forgetting." (Pre-empt with tight caliper matching + multiple tolerances.)
- "It only appears on synthetic data." (Pre-empt with Split-MNIST replication.)
- "Your method just takes smaller steps." (Pre-empt with matched-magnitude fairness control.)
- "The novelty is incremental over directions-of-curvature / catastrophic-overtraining." (Pre-empt by foregrounding the *simultaneous control + prospective controllability* that those lack.)

> 🟢 **In simple terms**
> The project scores well on depth, importance, and feasibility, and *reasonably* on novelty — its cleverness is in the careful isolation, not flashy new machinery. The paper's grade is a 7 today and only becomes an 8 if the key experiment actually finds the directional effect cleanly. The main ways a reviewer sinks it are all statistical/controls issues — which is exactly why the design front-loads cheap checks to catch them early.

---

<a name="21-paper-structure"></a>
## 21. Paper structure (target venue: top-tier ML)

1. **Introduction** — interference in continual learning is not only about magnitude; pose the directional-residual question.
2. **Related work** — magnitude/duration (catastrophic overtraining), directions of curvature (retrospective/scalar), null-space/orthogonal methods (past-facing), reset methods (plasticity). Position the gap.
3. **Problem formulation & descriptors** — define FTD, CTB, PTD, direction, and the directional residual precisely.
4. **Identifiability & decomposition** — confront multicollinearity head-on; describe the decorrelation-by-design generator and VIF gating.
5. **Controlled experiment (synthetic)** — Candidate B, with Control C front and center; the β₄ result and robustness.
6. **Intervention (method)** — Candidate A, matched-magnitude fairness control (if reached).
7. **Real-data replication** — Split-MNIST (and optionally Split-CIFAR).
8. **Limitations** — identifiability boundaries, generality of the directional descriptor, synthetic-to-real gap.
9. **Conclusion** — what the directional residual is, and what it implies for continual-learning method design.

> 🟢 **In simple terms**
> Tell the story in order: here's the overlooked question → here's what others did and didn't do → here are our careful definitions → here's how we made the statistics honest → here's the clean toy experiment → here's the method → here's proof it's not just a toy artifact → here's what we *can't* yet claim → here's what it means.

---

<a name="22-glossary"></a>
## 22. Glossary

- **Plasticity / loss of plasticity** — the network's ability to keep learning new tasks / the erosion of that ability over a task sequence.
- **Catastrophic forgetting** — degradation on *previously learned* tasks after learning new ones.
- **Future-Trainability Damage (FTD)** — how much a network's ability to learn a *fresh future* task is reduced by an update. The core quantity here.
- **Update vector Δθ / magnitude ‖Δθ‖ / direction d** — the change to the weights, its size, and its (size-free) direction.
- **Current-Task Benefit (CTB)** — how much the current task improved from the update.
- **Past-Task Damage (PTD)** — how much old tasks worsened (i.e., forgetting) from the update.
- **Directional residual (β₄)** — the effect of direction on FTD that survives after controlling for magnitude, current benefit, and past damage. The prize.
- **Multicollinearity / VIF** — when predictors are too correlated to separate their effects / a standard number quantifying how bad that is.
- **Control C** — matching past-task damage across compared updates; the most fragile control.
- **Teacher–student (synthetic)** — a setup where a student network learns to imitate a fixed random teacher; gives full control and unlimited data.
- **Split-MNIST / Split-CIFAR** — protocols that turn MNIST/CIFAR into a sequence of small tasks for continual-learning evaluation.
- **Backward Transfer (BWT) / ACC** — standard continual-learning metrics for forgetting and overall accuracy.
- **Kill condition** — a pre-committed criterion that ends or pivots the project, to prevent self-deception.
- **Pre-registration** — locking definitions and thresholds *before* seeing outcomes, so the hypothesis test stays honest.

---

*End of specification. The two immediate next artifacts to produce are (1) the frozen pre-registration document and (2) the Week-1 synthetic scaffold — and the single cheapest early test of viability is the Week 1–2 identifiability (VIF) pre-check, which can trigger kill condition #1 before any serious time is invested.*
