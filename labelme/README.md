# Mask Generation

1. run ```mask_generation.py``` to generate foot mask image
```
	python mask_generation.py --annotation_dir_name=<json directory name> --image_dir_name=<image directory name> --mask_dir_name=<output directory name>
	ex) python mask_generation.py --annotation_dir_name=annotation --image_dir_name=images --mask_dir_name=mask
```

2. run ```synthetic_image_mask_generator.py``` to generate synthetic images and masks (foot + coin)
```
	python synthetic_image_mask_generator.py --annotation_dir_name=<json directory name> --image_dir_name=<image directory name> --mask_dir_name=<mask directory name> --coin_dir_name=<coin directory name> --output_dir_name=<output directory name>
	ex) python mask_generation.py --annotation_dir_name=annotation --image_dir_name=images --mask_dir_name=mask --coin_dir_name=coin_images --output_dir_name=final_training_data
```
