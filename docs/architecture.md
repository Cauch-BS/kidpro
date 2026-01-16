# Architecture

KidPro has two training paths that share the same WSI inputs:

- Patch segmentation path: XML + SVS -> patches + masks -> tile model training.
- MIL path: WSI + labels -> tiled bags -> slide-level classifier training.

```mermaid
flowchart TD
  WSI[WSI_Slides] --> Preprocess[Preprocess_Tiles]
  Labels[Label_CSV] --> Preprocess
  Preprocess --> MILTiles[MIL_Tiles]
  MILTiles --> TrainWSI[Train_WSI_MIL]
  TrainWSI --> WSIModel[WSI_Model_Checkpoint]

  XML[XML_Annotations] --> PatchGen[Patch_Generation]
  WSI --> PatchGen
  PatchGen --> PatchDataset[Segmentation_Patches]
  PatchDataset --> TrainTile[Train_Tile_Seg]
  TrainTile --> TileModel[Tile_Model_Checkpoint]

  WSI --> Inference[WSI_Inference]
  WSIModel --> Inference
  Inference --> Prediction[Prediction_JSON]
```

Key components:

- `kidpro.patch`: generates image+mask patches from annotations.
- `kidpro.preprocessing`: creates MIL tiles from WSIs.
- `kidpro.train_tile`: trains segmentation on patches.
- `kidpro.train_wsi`: trains WSI MIL classifier on tiles.
- `kidpro.infer_wsi`: tiles a WSI (if needed) and runs classification.

See also: `overview.md`, `training.md`, `inference.md`.
