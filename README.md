# Rank Induction for Multiple Instance Learning

Official repository for:
Kim et al., "Ranking-Aware Multiple Instance Learning for Histopathology Slide Classification"

Overview
Rank Induction is a training strategy for Multiple Instance Learning (MIL) that leverages expert annotations in a more flexible manner than traditional attention-based approaches. By ranking annotated lesion patches higher than non-lesion patches, our method guides the model to focus on diagnostically meaningful regions without over-constraining attention distribution.
## 🧠 Overview

**Rank Induction** is a training strategy for Multiple Instance Learning (MIL) that leverages expert annotations using a **ranking constraint**—rather than exact attention matching—to guide the model’s focus toward diagnostically meaningful areas.

> ⚠️ Most MIL methods either ignore expert annotations or enforce overly strict attention constraints.  
> ✅ Our method strikes a balance by ranking annotated lesion patches higher than non-lesion ones—offering better interpretability and performance, especially in low-data regimes.

---

## 🔬 Method

### 1. Ranking Constraint

For each annotated patch \( i \) and non-annotated patch \( j \), we enforce:
```math
s_i > s_j
```
We convert the score difference to a pairwise ranking probability:
```math
P_{i,j} = \frac{1}{1 + e^{-\sigma (s_i - s_j - m)}}
```