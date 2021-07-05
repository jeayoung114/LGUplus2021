# LGUplus2021
Framework on Recommender Systems

## Available models
> For explicit feedback (e) <br>
> For implicit feedback (i)
---------------------------------------

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
  - EASE (i)

- Neural item-item collaborative filtering
  - AutoRec (i)
  - DAE (i)
  - CDAE (i)
  - MultVAE (i)
