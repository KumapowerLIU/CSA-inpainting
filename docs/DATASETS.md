# Training data

CSA learns from **complete RGB images**. The training loader creates the missing regions, so you do not need paired damaged images, text captions, class labels, or audio. You can train on your own image collection using the folder-to-index recipe in [Training](TRAINING.md#prepare-independent-runtimes).

This page describes the data used in the **modern step-97,397 release** and where to obtain the source datasets. It is an experiment record for this branch, not the dataset protocol of the ICCV 2019 paper.

## What the released model used

| Source | Training contribution | Purpose |
|---|---:|---|
| Existing Unsplash-derived collection | 296,145 indexed images | General photographs |
| Places365-Standard | 901,730 images, half of the 1,803,460-image training archive | Indoor and outdoor scenes |
| HDTF frames, added during stage 2 | 30,128 selected frames represented by 299,469 sampling entries | More face examples |

The Unsplash + Places mixture contains **1,197,875 training entries**. After adding HDTF, the final stage-2 index contains **1,497,344 entries**. Each selected HDTF frame appears 9 or 10 times, making faces approximately **20% of sampling entries**. These totals describe the sampling index; they are not counts of unique image content. No cross-dataset content deduplication was performed.

The training lineage started with Unsplash, added the fixed Places subset during stage 1, and later added HDTF during stage 2. The face mixture therefore does not describe every historical update of either stage. The released rough-network parent is stage-1 step 156,181; the refinement checkpoint is stage-2 step 97,397. See [checkpoint provenance](CHECKPOINTS.md).

### Validation boundaries

- The general validation set contains **2,000 held-out Unsplash records**. It stayed unchanged when Places and faces were added.
- HDTF has a separate **455-frame holdout from 20 source-video groups**. Splitting used source-video grouping; identity-disjointness has not been established.
- The fixed 64-image progress evaluations use the general Unsplash holdout. They are not a dedicated face benchmark or a Places benchmark. The release preview shows a smaller qualitative selection; see the [evaluation record](releases/step97397/README.md).

## Where to get the data

### Unsplash: general photographs

Start with the [official Unsplash Dataset page](https://unsplash.com/data) and [dataset repository](https://github.com/unsplash/datasets). The repository provides a Lite download and instructions for requesting Full access. Its [schema documentation](https://github.com/unsplash/datasets/blob/master/DOCS.md) describes `photos.tsv`, including the `photo_image_url` field: the dataset metadata is not a ready-made folder of training JPEGs. Follow the publisher's access instructions to obtain the images, then create your own train/validation folders.

The local collection used here was already stored in custom TFRecord shards. Its 296,145/2,000 split and original image selection are not distributed. We have not established that it corresponds to a particular official Lite/Full version. Downloading a current official Unsplash release will therefore **not reproduce this exact training split**.

Use the publisher's [dataset terms](https://github.com/unsplash/datasets/blob/master/TERMS.md) for the version you obtain. The Full dataset has non-commercial restrictions; do not infer image or model rights from this code repository's license.

### Places365-Standard: scene photographs

Use the [Places download page](http://places2.csail.mit.edu/download.html), linked from the [official CSAILVision repository](https://github.com/CSAILVision/places365). Choose **Places365-Standard training images**. Standard contains about 1.8 million training images in 365 categories; Places365-Challenge is a different, larger collection. The official repository also links an easy-format archive for ordinary image-folder loading. If the dataset server is unavailable, check the official repository for current access information.

This experiment indexed the existing **uncompressed** `train_large_places365standard.tar` archive directly. It selected 901,730 of the archive's 1,803,460 regular JPEG members with:

```python
selected_ordinals = set(
    random.Random(20261023).sample(range(1_803_460), 901_730)
)
```

Ordinals follow **JPEG member order inside the TAR**, not sorted filenames. The same seed with a repacked archive can select different images. This is a random half of the full training archive, not an exact half of every class.

The implementation is [prepare_places_mix.py](../scripts/remote/prepare_places_mix.py). It records member names, byte offsets, lengths, and selection hashes, and keeps the existing general validation index unchanged. It expects the legacy Unsplash index described above; it is not a general image-folder converter. For a fresh public-data experiment, extract your chosen images and use the ordinary folder recipe in [Training](TRAINING.md#prepare-independent-runtimes).

Places images remain subject to their owners' rights and the publisher's data terms. The Places CNN code/model license is not a blanket license for every training image.

### HDTF: talking-head frames

Use the [official HDTF repository](https://github.com/MRzzm/HDTF). Its `HDTF_dataset` directory provides source video URLs, clip timestamps, and crop information. The README explains downloading the videos, making talking-head clips, cropping faces, and resizing to **512 × 512**. Follow that preprocessing and extract still frames for CSA; the audio branch of HDTF is unnecessary here. The authors identify the dataset as CC BY 4.0 in their README, separately from the repository's code license.

Our experiment reused an existing 512 × 512 frame extraction. The available source contained 409 clips grouped into 345 source videos. Training selection used uniform frame-order bins, at least 25 frame positions between selections, and at most 100 selected frames per training clip. This yielded 30,128 frames. Sampling repetitions produced the approximately 20% face contribution described above.

This selection and holdout are local experiment choices, not HDTF's official benchmark split. The private frame manifests and exact pre-existing extraction are not included. A newly downloaded and processed HDTF collection may differ in available videos, crop pixels, and frame indices.

## Prepare a new training run

1. Obtain one or more image sources through their official channels. All three datasets are optional: an existing RGB image collection is enough to start.
2. Create separate training and validation folders. Keep related video frames together when splitting HDTF, and check image-content overlap across sources.
3. Follow [Training](TRAINING.md) to build `train.jsonl`, `validation.jsonl`, and `dataset.json`, then train the rough network before the refinement network.

For example, the image-folder manifest recipe accepts:

```text
data/
├── train/
│   ├── photographs/
│   ├── scenes/
│   └── faces/
└── validation/
    ├── photographs/
    ├── scenes/
    └── faces/
```

Folder names are arbitrary; only image contents and the supplied split matter. The example recipe lists each image once. It does **not** automatically recreate the release's Places subset, face weighting, or private validation split.

The release trains at **256 × 256** with horizontal flips and online random masks: rectangles, free-form strokes, or their union, covering 10%–40% of an image. No external mask dataset is required. These generated masks are not the original paper's irregular-mask benchmark. See [masks.py](../src/csa_inpainting/masks.py) and the [release configuration](releases/step97397/training-config.yaml).

The public release includes code, model weights, configuration, and evaluation provenance. Full training datasets and private training indexes are not redistributed.
