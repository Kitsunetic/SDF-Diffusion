# SDF-Diffusion




Diffusion-Based Signed Distance Fields for 3D Shape Generation (CVPR 2023)

[**Paper**](https://openaccess.thecvf.com/content/CVPR2023/html/Shim_Diffusion-Based_Signed_Distance_Fields_for_3D_Shape_Generation_CVPR_2023_paper.html) | [**Project Page**](https://kitsunetic.github.io/sdf-diffusion/)




## Requirements

- pytorch
- pytorch3d
- h5py
- einops
- scipy
- scikit-image
- tqdm
- point-cloud-utils




## Dataset

The preprocessed dataset can be downloaded in [Huggingface](https://huggingface.co/datasets/kitsunetic/SDF-Diffusion-Dataset)

The dataset (~13GB for resolution 32, ~50GB for 64) should be unzipped and located like this:

```
SDF-Diffusion
├── config
    ├── gen32
        ├── airplane.yaml
        ├── ...
        ├── shapenet.yaml
    ├── sr32_64
        ├── airplane.yaml
        ├── ...
        ├── shapenet.yaml
├── src
    ├── datasets  # dataset-related codes
    ├── models  # network architectures
    ├── utils
    ├── ...
    ├── trainer.py  # custom trainer
├── results  # pretrained checkpoints
    ├── gen32
        ├── airplane.pth
        ├── ...
        ├── shapenet.pth
    ├── sr32_64
        ├── airplane.pth
        ├── ...
        ├── shapenet.pth        
├── main.py

data
├── sdf.res32.level0.0500.PC15000.pad0.20.hdf5
├── sdf.res64.level0.0313.PC15000.pad0.20.hdf5
```

Before downloading the dataset, please create [ShapeNet webpage](https://shapenet.org) and consider citing ShapeNet:
```bib
@article{chang2015shapenet,
  title={Shapenet: An information-rich 3d model repository},
  author={Chang, Angel X and Funkhouser, Thomas and Guibas, Leonidas and Hanrahan, Pat and Huang, Qixing and Li, Zimo and Savarese, Silvio and Savva, Manolis and Song, Shuran and Su, Hao and others},
  journal={arXiv preprint arXiv:1512.03012},
  year={2015}
}
```
The dataset can be used only for non-commercial research and educational purpose.




## Demo

You can download pretrained checkpoints for [unconditional](https://github.com/Kitsunetic/SDF-Diffusion/releases/download/checkpoint/gen32.zip) and [category-conditional](https://github.com/Kitsunetic/SDF-Diffusion/releases/download/checkpoint/sr32_64.zip).
Please unzip the `.zip` files in `./results` folder.

You can find demo scripts in [unconditional](./scripts/demo-single_category.ipynb) and [category-conditional](./scripts/demo-multi-category.ipynb).




## Training

### Single Category Unconditional Generation

```sh
# generation (resolution 32)
python main.py config/gen32/{airplane|car|chair}.yaml

# super resolution (resolution 32 -> 64)
python main.py config/sr32_64/{airplane|car|chair}.yaml
```

### Category Conditional Generation

```sh
# generation (resolution 32)
python main.py config/gen32/shapenet.yaml

# super resolution (resolution 32 -> 64)
python main.py config/sr32_64/shapenet.yaml
```


<!-- 

## Inference & Evaluation

### Pretrained Models

TBD

### Unconditional Generation

TBD

### Category-Conditioned Generation

TBD

 -->


## Citation

```bib
@inproceedings{shim2023diffusion,
  title={Diffusion-Based Signed Distance Fields for 3D Shape Generation},
  author={Shim, Jaehyeok and Kang, Changwoo and Joo, Kyungdon},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20887--20897},
  year={2023}
}
```
