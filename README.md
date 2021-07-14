# LGUplus2021
Framework on Recommender Systems

## Available models
> For explicit feedback (e) <br>
> For implicit feedback (i)
<!-- --------------------------------------- -->

- Memory-based collaborative filtering
  - UserKNN (e, i)
  - ItemKNN (e, i)

- Model-based collaborative filtering
  - SVD (e, i)
  - MF (e)
  - LogisticMF (i)
  - WMF w/ ALS (i)
  - WMF w/ GD (i)

- Linear item-item collaborative filtering
  - SLIM (i)
  - EASE<sup>R</sup> (i)

- Neural item-item collaborative filtering
  - U-AutoRec (i)
  - I-AutoRec (i)
  - DAE (i)
  - CDAE (i)
  - MultVAE (i)

- Neural collaborative filtering
  - GMF (i)
  - MLP (i)
  - NeuMF (i)

- Graph collaborative filterig
  - NGCF (i)
  - LightGCN (i)

## How to run
1. Edit models' hyper-parameters you want in ```main.py```
2. Run ```main.py```
