# Images for README.md

Please place the following images from the paper into the `imgs/` folder with these exact filenames:

## Required Images

1. **darenet_architecture.png**
   - Description: Main DARE-Net architecture diagram showing the complete pipeline
   - From: Figure showing the full model architecture with ACDense backbone, MoE routing, and multi-task heads
   - Size: Recommended 1200x800px or similar

2. **modular_pipeline.png**
   - Description: Modular brain age model pipeline diagram
   - From: Figure showing the modular pipeline with shared encoder, staging head, and Top-K router
   - Size: Recommended 1000x600px or similar

3. **stage_heterogeneity.png**
   - Description: Stage-dependent heterogeneity visualization
   - From: Figure showing different trajectories for CN/MCI/AD groups
   - Size: Recommended 800x600px or similar

4. **teacher_forcing.png**
   - Description: Scheduled teacher forcing strategy diagram
   - From: Figure showing the transition from ground-truth labels to predicted posteriors
   - Size: Recommended 800x400px or similar

5. **clinical_interpretation.png**
   - Description: Clinical interpretation pipeline (BAG → BAGZ → Risk stratification)
   - From: Figure showing the clinical interpretation workflow
   - Size: Recommended 1000x600px or similar

6. **evaluation_results.png**
   - Description: Comprehensive evaluation results (6-panel figure)
   - From: Figure showing age prediction scatter plots, confusion matrix, BAG distribution, etc.
   - Size: Recommended 1800x1200px or similar (large figure)

7. **mae_by_age.png**
   - Description: MAE comparison by age bins (grouped bar chart)
   - From: Figure comparing DARE-Net with baselines across age groups
   - Size: Recommended 1000x600px or similar

8. **gender_subgroup.png**
   - Description: Gender subgroup MAE scatter plot
   - From: Figure showing MAE for Gender=0 vs Gender=1
   - Size: Recommended 800x600px or similar

9. **tsne_embedding.png**
   - Description: t-SNE visualization of shared embedding space
   - From: Figure showing t-SNE plot with CN/MCI/AD groups
   - Size: Recommended 1000x800px or similar

10. **gradcam_attention.png**
    - Description: Average Grad-CAM attention maps
    - From: Figure showing attention patterns across brain slices
    - Size: Recommended 1200x800px or similar

11. **bag_analysis.png**
    - Description: Brain Age Gap (BAG) analysis (4-panel figure)
    - From: Figure showing BAG groups, AD prevalence, odds ratios
    - Size: Recommended 1600x1200px or similar (large figure)

## Optional Images (for detailed documentation)

- `acdense_backbone.png` - ACDense backbone architecture detail
- `moe_routing.png` - Detailed MoE routing mechanism
- `results_comparison.png` - Detailed comparison with more baselines
- `uncertainty_calibration.png` - Uncertainty calibration plots

## Image Format

- **Format**: PNG (preferred) or JPG
- **Resolution**: High resolution (at least 300 DPI for print quality)
- **Background**: White or transparent (for PNG)
- **Naming**: Use lowercase with underscores (snake_case)

## Notes

- All images should be from the published paper
- Ensure images are clear and readable at the sizes displayed in README
- If images are too large, consider creating optimized versions for web display
- Maintain aspect ratios when resizing

