# CSA pipeline illustration

This is an editable vector schematic of the **modern two-stage implementation**.
It was drawn directly as SVG, using Figure 2 of the [ICCV 2019 paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Coherent_Semantic_Attention_for_Image_Inpainting_ICCV_2019_paper.pdf) as the visual reference. The drawing uses thin feature-map plates, dashed U-Net skip connections, a gray rough network, red/blue refinement layers, and a yellow CSA layer. It has no generated artwork, gradients, shadows, or decorative cards.

## Source and export

- `csa-pipeline-paper.svg` is the editable source (1840 × 650 viewBox).
- `csa-pipeline-paper.png` is its 2× raster export (3680 × 1300), used where SVG rendering is unavailable.
- The SVG embeds the existing `examples/masked.png`, `examples/mask.png`, and `examples/expected.png` without changing their pixels. The right image is the released model's actual output, not a ground-truth target or a generated illustration.

## Implementation mapping

The graph was checked against `Stage2InpaintingModel.forward` and `RefinementNetwork.forward`:

1. The rough network receives the masked RGB input.
2. A coarse composite combines rough predictions inside the hole with the known input pixels.
3. Refinement receives the coarse composite concatenated with the masked input.
4. CSA operates at the 32 × 32 encoder level for 256 × 256 inputs; it is not the 1 × 1 bottleneck.
5. Final compositing retains known input pixels and uses refinement predictions inside the hole.

Feature-map plates denote layer groups; they are not a complete operator-by-operator specification. The bottom notes describe the modern sequential training recipe: rough is trained first, then frozen during refinement training. LPIPS is a modern addition. The original paper already includes reconstruction, consistency, and relativistic average least-squares GAN supervision.

The schematic follows the paper's visual conventions; it is not a reproduction of its original figure or a claim that this release exactly reproduces the paper's training protocol.
