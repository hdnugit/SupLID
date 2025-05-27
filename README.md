# SupLID
SupLID: Geometrical Guidance for Out-of-Distribution Detection in Semantic Segmentation

This is the official implementation of the [SupLID: Geometrical Guidance for Out-of-Distribution Detection in Semantic Segmentation]().


## Package installation
* python                    3.11.4
* pytorch-cuda              11.8 
* torchvision               0.15.2
* torch-cluster             1.6.1+pt20cu118
* torch-geometric           2.3.1


## Data_preprocessing

```
python data_preprocessing.py --ckpt_path <path_to_segmentation_ckpt> --op <output_file_path>
```

## Inference

```
python inference.py --root <path_to_preprocessed_data>
```



## Acknowledgement

Part of our codes are adapted from below repo:

lid_adversarial_subspace_detection - https://github.com/xingjunm/lid_adversarial_subspace_detection.git - Apache-2.0 license


## Citation
```

```

## Security



## License

This project is licensed under the '' License.
