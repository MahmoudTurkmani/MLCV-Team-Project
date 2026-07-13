#!/bin/bash

heads=(linear mlp)
backbones=(Base Large Huge)
augments=("" --pink_noise --white_noise --pitch_shift --mixcut)

for backbone_size in "${backbones[@]}"; do
  for augment in "${augments[@]}"; do
    for head in "${heads[@]}"; do
      echo sbatch slurm-birb.slurm --head "$head" --backbone_size "$backbone_size" "$augment"
    done
  done
done
