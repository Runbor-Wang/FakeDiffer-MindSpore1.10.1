import numpy as np
from os import listdir
from os.path import join
from glob import glob
from dataset import AbstractDataset

SPLITS = ["train", "test"]


class WildDeepfake(AbstractDataset):
    """
    Wild Deepfake Dataset proposed in "WildDeepfake: A Challenging Real-World Dataset for Deepfake Detection"
    """
    def __init__(self, cfg):
        super(WildDeepfake, self).__init__(cfg)
        # pre-check
        # self.splits = cfg['split']
        if cfg['split'] not in SPLITS:
            raise ValueError(f"split should be one of {SPLITS}, but found {cfg['split']}.")
        # super(WildDeepfake, self).__init__(cfg, seed, transforms, transform, target_transform)
        print(f"Loading data from 'WildDeepfake' of split '{cfg['split']}'"
              f"\nPlease wait patiently...")
        self.categories = ['real', 'fake']
        self.root = cfg['root']
        images_ids = self.__get_images_ids()
        train_ids = [images_ids[0], images_ids[2]]
        test_ids = [images_ids[1], images_ids[3]]
        self.images, self.targets = self.__get_images(
            [test_ids, "test"] if cfg['split'] == "test" else [train_ids, "train"], cfg['balance'])
        assert len(self.images) == len(self.targets), "The number of images and targets not consistent."
        print("Data from 'WildDeepfake' loaded.\n")
        print(f"Dataset contains {len(self.images)} images.\n")

    def __get_images_ids(self):
        real_train = listdir(join(self.root, 'real_train'))
        real_test = listdir(join(self.root, 'real_test'))
        fake_train = listdir(join(self.root, 'fake_train'))
        fake_test = listdir(join(self.root, 'fake_test'))
        return set(real_train), set(real_test), set(fake_train), set(fake_test)

    def __get_images(self, ids, balance=False):
        real = list()
        fake = list()
        # real
        if ids[1] == "train":
            for _ in ids[0][0]:
                real.extend(glob(join(self.root, 'real_train', _, '*.png')))
            for _ in ids[0][1]:
                fake.extend(glob(join(self.root, 'fake_train', _, '*.png')))
        else:
            for _ in ids[0][0]:
                real.extend(glob(join(self.root, 'real_test', _, '*.png')))
            for _ in ids[0][1]:
                fake.extend(glob(join(self.root, 'fake_test', _, '*.png')))

        if balance:
            fake = np.random.choice(fake, size=len(real), replace=False)
            print(f"After Balance | Real: {len(real)}, Fake: {len(fake)}")
        real_tgt = [0] * len(real)
        fake_tgt = [1] * len(fake)
        return [*real, *fake], [*real_tgt, *fake_tgt]

    def __get_test_ids(self):
        youtube_real = set()
        celeb_real = set()
        celeb_fake = set()
        with open(join(self.root, "List_of_testing_videos.txt"), "r", encoding="utf-8") as f:
            contents = f.readlines()
            for line in contents:
                name = line.split(" ")[-1]
                number = name.split("/")[-1].split(".")[0]
                if "YouTube-real" in name:
                    youtube_real.add(number)
                elif "Celeb-real" in name:
                    celeb_real.add(number)
                elif "Celeb-synthesis" in name:
                    celeb_fake.add(number)
                else:
                    raise ValueError("'List_of_testing_videos.txt' file corrupted.")
        return youtube_real, celeb_real, celeb_fake

    def __getitem__(self, index):
        path = join(self.root, self.split, self.images[index])
        tgt = self.targets[index]
        return path, tgt
        # return self.load_item(path), tgt


if __name__ == '__main__':
    import yaml

    config_path = "../config/dataset/wilddeepfake.yml"
    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    config = config["train_cfg"]
    # config = config["test_cfg"]


    def run_dataset():
        dataset = WildDeepfake(config)
        print(f"dataset num: {len(dataset)}")
        for i, _ in enumerate(dataset):
            path, target = _
            print(f"path: {path}, target: {target}")
            if i >= 9:
                break


    # def run_dataloader(display_samples=False):
    #     from torch.utils import data
    #     import matplotlib.pyplot as plt
    #
    #     dataset = WildDeepfake(config)
    #     dataloader = data.DataLoader(dataset, batch_size=8, shuffle=True)
    #     print(f"dataset: {len(dataset)}")
    #     for i, _ in enumerate(dataloader):
    #         path, targets = _
    #         image = dataloader.dataset.load_item(path)
    #         print(f"image: {image.shape}, target: {targets}")
    #         if display_samples:
    #             plt.figure()
    #             img = image[0].permute([1, 2, 0]).numpy()
    #             plt.imshow(img)
    #             # plt.savefig("./img_" + str(i) + ".png")
    #             plt.show()
    #         if i >= 9:
    #             break


    ###########################
    # run the functions below #
    ###########################

    run_dataset()
    # run_dataloader(False)
