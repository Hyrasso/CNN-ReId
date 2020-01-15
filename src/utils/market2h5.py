import h5py
from keras.preprocessing.image import load_img, img_to_array
import json
from pathlib import Path
import time
import numpy as np

MARKET_IMAGE_DIMS = (128, 64, 3) # height x width x channels
LABEL_DIM = 27
def market_to_h5(folder: str, attribute: str, destination: str):
    """Create an hdf5 file from image folder and JSON labels file
    
    The HDF5 file contains 4 datasets at the root:
        - train_images and test_images conataining the images, shape: N*Height*Width*Channels
        - train_labels and test_labesl containing the labels values, and their names in attr, shape: N*#labels 
    """
    # 1 dataset / train_images
    # 1 dataset / train_labels
    
    # 1 dataset / test_images
    # 1 dataset / test_labels
    start = time.time()
    folder = Path(folder)
    assert folder.is_dir(), "Valide folder path"
    attribute = Path(attribute)
    assert attribute.is_file(), "Valid json file path"
    with h5py.File(destination, "w") as hf, open(attribute) as meta:
        meta_data = json.load(meta)
        for split_name in ("train", "test"):
            split = meta_data[split_name]
            attrs = sorted([a for a in split if a != "image_index"])
            assert len(attrs) == LABEL_DIM, "Number of labels"
            print("Processing", split_name, "data")

            images = [f for f in folder.glob(f"*.jpg") if f.name[:4] in split["image_index"]]

            xds = hf.create_dataset(f"{split_name}_images", (len(images), *MARKET_IMAGE_DIMS))
            yds = hf.create_dataset(f"{split_name}_labels", (len(images), LABEL_DIM))
            for j, image in enumerate(images):
                if j%10==0:print("\r",round(j / len(images) * 100), "%", end=" ")
                idx = split["image_index"].index(image.name[:4])
                # no resize/preprocessing, 
                # image data is written as it is given by keras preprocessing functions
                im = img_to_array(load_img(image))
                xds[j] = im
                yds[j] = np.array([split[attr][idx] for attr in attrs])
                yds.attrs["labels"] = attrs
            print()
    print(f"HDF5 file created at {destination} in {round(time.time() - start)} sec")

if __name__ == "__main__":
    market_to_h5("Market-1501", "market_attribute.json", "market.h5")