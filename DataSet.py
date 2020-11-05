from torch.utils.data import DataLoader, Dataset
import torch
class WheatDataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms = None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms
    def __getitem__(self, index: int):

        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]
        #image = plt.imread(f'{self.image_dir}/{image_id}.jpg')
        image = image = cv2.imread(f'{self.image_dir}/{image_id}.jpg') # reads the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to grey scale
        image = cv2.resize(image, (256,256))
        norm_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #print(norm_image[190:200, 170:200])
        image = np.array([norm_image])
        blanc = np.zeros((2, 32, 32)).astype(np.float32)
        for _, row in self.df[self.df["image_id"] == image_id].iterrows():
            #blanc[ :, row.bbox_ymin//4 : (row.bbox_ymin + row.bbox_height)//4, row.bbox_xmin//4 : (row.bbox_xmin + row.bbox_width)//4] = 1
            Y =  row.bbox_ymin + row.bbox_height // 2
            X = row.bbox_xmin + row.bbox_width // 2
            blanc[0][Y//32][X//32] = row.bbox_height/32
            blanc[1][Y//32][X//32] = row.bbox_width/32

        """if self.transforms:
          image = self.transforms(image)"""

        return torch.from_numpy(image), torch.from_numpy(blanc)

    def __len__(self) -> int:
        return self.image_ids.shape[0]
