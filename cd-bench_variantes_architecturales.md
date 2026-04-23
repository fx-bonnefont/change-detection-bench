# cd-bench — Variantes architecturales à benchmarker

**Contexte** : extension du projet cd-bench (semantic change detection sur HI-UCD avec backbone DINOv2) pour explorer des architectures de fusion temporelle inspirées de Flamingo.

**Hypothèse de recherche** : remplacer la fusion tardive (différence ou concaténation) par une fusion progressive via cross-attention gatée intercalée dans le backbone permet de mieux capturer les changements à différentes échelles sémantiques, tout en préservant les features d'un foundation model gelé.

---

## 1. Baseline de référence

### V0 — Siamois DINOv2 + différence (baseline obligatoire)
- Deux branches DINOv2 partagées, gelées
- Features T1 et T2 extraites indépendamment
- Fusion par différence absolue : `|f_T1 - f_T2|`
- Tête de segmentation légère (UPerNet ou simple FPN)
- **Rôle** : point de comparaison absolu, déjà implémenté dans cd-bench

---

## 2. Variantes "fusion tardive" (baselines complémentaires)

### V1 — Siamois DINOv2 + concaténation
- Identique à V0, fusion par `concat([f_T1, f_T2])`
- Tête adaptée pour gérer le doublement de channels
- **Objectif** : valider que la différence est bien le bon opérateur de fusion

### V2 — Siamois DINOv2 + cross-attention finale
- Une couche de cross-attention bidirectionnelle après extraction des features
- T1 attend T2 et inversement, fusion des deux sorties
- **Objectif** : baseline "attention" honnête, proche de l'esprit BIT/ChangeFormer

---

## 3. Variantes "fusion progressive" (contribution principale)

### V3 — XATTN gatée intercalée, asymétrique ⭐
- DINOv2 traite T1 (branche principale)
- Features T2 extraites une fois en amont par DINOv2 gelé
- Couches `GATED XATTN-DENSE` insérées entre N blocs du ViT principal
- Gating tanh initialisé à 0 (style Flamingo)
- Formule : `output = x + tanh(α) · XATTN(x, features_T2)` puis FFN gaté
- **Rôle** : architecture cœur de la contribution

### V4 — XATTN gatée intercalée, symétrique (deux passes)
- Deux forwards : T1 conditionné par T2, puis T2 conditionné par T1
- Fusion des deux sorties en fin de pipeline
- 2× plus coûteux en compute mais respecte la symétrie naturelle de SCD
- **Objectif** : adresser la critique d'asymétrie de V3

### V5 — Co-attention gatée intercalée, symétrique (une passe)
- Deux branches DINOv2 traitées en parallèle
- À chaque niveau d'insertion, cross-attention bidirectionnelle gatée entre les branches
- Une seule passe mais mémoire 2× (les deux branches actives simultanément)
- **Objectif** : version élégante de V4, meilleur compromis perf/coût

---

## 4. Ablations sur le mécanisme de gating

### V6 — XATTN intercalée SANS gating
- Identique à V3 mais sans le facteur `tanh(α)`
- Cross-attention résiduelle classique
- **Objectif** : isoler la contribution du gating à la stabilité d'entraînement et à la performance

### V7 — XATTN intercalée, gate initialisé à valeur non-nulle
- Identique à V3 mais α initialisé à 0.1 ou 0.5 au lieu de 0
- **Objectif** : valider l'importance du démarrage à zéro (préservation initiale du backbone)

---

## 5. Ablations sur la position des couches XATTN

À combiner avec V3 :

| Variante | Insertion | Hypothèse testée |
|---|---|---|
| **V3a** | Premiers blocs uniquement | Early fusion : capter le changement bas-niveau (texture, couleur) |
| **V3b** | Derniers blocs uniquement | Late fusion progressive : capter le changement sémantique haut-niveau |
| **V3c** | Tous les blocs | Max fusion, max compute |
| **V3d** | Tous les 2-3 blocs | Compromis perf/compute, probablement optimal |

L'ablation de profondeur est souvent ce qui rend une contribution architecturale convaincante.

---

## 6. Variantes "économie de compute"

### V8 — V3 + Perceiver Resampler sur T2
- Compresse les patch tokens de T2 à un nombre fixe (32 ou 64) via Perceiver Resampler
- Réduit drastiquement le coût de cross-attention
- Pertinent pour passer à des tuiles satellite haute résolution

### V9 — V3 avec LoRA sur les XATTN
- Entraîner les XATTN en LoRA (rank 8 ou 16) plutôt qu'en full fine-tuning
- Encore plus léger en paramètres entraînables
- **Objectif** : tableau "params trainable vs perf"

---

## 7. Plan de benchmark recommandé pour le mémoire

### Priorisation

| Priorité | Variantes | Justification |
|---|---|---|
| 🔴 **Must** | V0, V2, V3 | Baselines + contribution principale |
| 🟠 **Should** | V5, V6 | Symétrie + ablation gating (cruciaux à l'oral) |
| 🟢 **Nice** | V3c vs V3d, V8 | Profondeur d'insertion + dimension efficacité |
| ⚪ **Optional** | V1, V4, V7, V9 | Si temps et compute disponibles |

### Plan minimal défendable (5 variantes)
**V0, V2, V3, V5, V6** + une ablation de profondeur sur V3.

Ce sous-ensemble couvre :
- Baseline siamoise classique
- Baseline attention "honnête"
- Contribution principale (Flamingo-style)
- Réponse à la critique de symétrie
- Ablation du gating (défense du choix architectural)

---

## 8. Métriques à reporter

Pour chaque variante, sur HI-UCD :

### Métriques de performance
- **mIoU** (binary change detection)
- **SeK** (semantic change)
- **F1scd**
- **Précision / Rappel** par classe sémantique

### Métriques d'efficacité
- Nombre de **paramètres entraînables**
- Nombre de **paramètres totaux**
- **Temps d'entraînement** par epoch
- **Mémoire GPU peak** (training)
- **Temps d'inférence** par image (à isoler)
- **VRAM inférence**

### Robustesse statistique
- **3 seeds minimum** par expérience
- Reporter **moyenne ± écart-type**
- Test de significativité (Wilcoxon ou t-test apparié) entre V3 et V0

---

## 9. Considérations d'implémentation

### Coût mémoire de la cross-attention
- Complexité : `O(N_T1 × N_T2 × d)` où N est le nombre de patch tokens
- Pour DINOv2 ViT-L/14 sur 224×224 : N ≈ 256 tokens → gérable
- Pour 448×448 ou plus : envisager Perceiver Resampler (V8)

### Initialisation
- Gate `α` : tenseur scalaire par couche, init à `0.0`, requires_grad=True
- XATTN weights : init standard (Xavier ou Kaiming), comme dans MultiheadAttention PyTorch

### Stack technique cd-bench
- PyTorch Lightning (déjà en place)
- MLFlow tracking sur EC2 (déjà en place)
- `uv` pour la gestion d'environnement (déjà en place)
- Backend MPS (MacBook Pro M4 Pro) pour debug local, EC2 GPU pour runs sérieux

---

## 10. Stratégie pour le mémoire

### Pour un Mastère Spécialisé : privilégier la rigueur
Mieux vaut **5 variantes bien benchmarkées avec 3 seeds** qu'un tableau de 10 lignes sans variance.

### Structure défendable du chapitre architectural
1. Motivation : limites de la fusion tardive en SCD
2. Inspiration Flamingo : gated cross-attention pour foundation models gelés
3. Présentation des V0, V2, V3, V5
4. Ablation V6 (gating) et V3a-d (profondeur)
5. Tableau perf vs efficacité
6. Discussion : symétrie, scaling, transférabilité

### Risques à anticiper
- ⚠️ V3 peut sous-performer V0 si le dataset est trop petit pour entraîner les XATTN
- ⚠️ Si toutes les variantes sont équivalentes en mIoU, défendre la contribution sur l'efficacité (V9 + LoRA)
- ⚠️ Comparer à au moins une SOTA récente (ChangeFormer, ScratchFormer) pour la crédibilité

---

## 11. Pistes d'extension (post-mémoire)

- Transposition à des séries temporelles longues (T1, T2, T3, ..., Tn) — naturellement géré par le mécanisme XATTN
- Conditionnement multi-modal : injecter aussi des données radar (Sentinel-1) en plus de l'optique
- Pré-entraînement self-supervised de l'architecture sur paires temporelles non annotées
- Distillation de V5 vers V3 pour réduire le coût d'inférence
