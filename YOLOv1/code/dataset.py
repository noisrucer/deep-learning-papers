import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os

class VOCDataset(Dataset):
    def __init__(self, data_csv, img_dir, label_dir, S=7, B=2, C=20, transform=None):
        '''
        Parameters:
            data_csv (str): csv file path name which has two columns.
                           1st column: img file name
                           2nd column: label file name
            img_dir (str): image directory path
            label_dir (str): label directory path
            S (int): Grid cell size (ex. 7x7)
            B (int): Number of bounding boxes per one cell
            C (int): Number of classes
            transform (Compose): A list of transforms
        '''

        self.df = pd.read_csv(data_csv)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform


    def __len__(self):
        '''
        Returns the length of the dataset
        '''
        return len(self.df)


    def __getitem__(self, idx):
        '''
        Returns one data item with idx

        Parameters:
            idx (int): Index number of data

        Returns:
            tuple: (image, label) where label is in format of
                    (S, S, 30). Note that last 5 elements of label
                    will NOT be used.
        '''

        # Path for image and label
        img_path = os.path.join(self.img_dir, self.df.iloc[idx, 0])
        label_path = os.path.join(self.label_dir, self.df.iloc[idx, 1])

        # Read image
        image = Image.open(img_path)

        # Read bounding boxes from label .txt file and store them to list
        bboxes = []

        with open(label_path, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break

                # class_num, x, y, w, h
                one_box = line.replace('\n','').split(' ')

                # convert types to int or float accordingly
                one_box = [
                    float(el) if float(el) != int(float(el)) else int(el)
                    for el in one_box
                           ]
                bboxes.append(one_box)

        bboxes = torch.tensor(bboxes)
        if self.transform:
            image, bboxes = self.transform(image, bboxes)
        bboxes = bboxes.tolist()

        '''
        Loop through each box and scale relative to grid cell.
        Formula is quite straightforward.
        cell_w = S * w
        cell_h = S * h
        cell_x = S * x - floor(x * S)
        cell_y = S * y - floor(y * S)
        '''

        label_matrix = torch.zeros((self.S, self.S, 30)) # last 5 will NOT be used!
        for box in bboxes:
            # (i,j) in SxS grid -> used to assign box to label_matrix
            j, i = int(box[1] * self.S), int(box[2] * self.S)

            box_class = int(box[0])

            # Rescale relative to cell
            box[1] = self.S * box[1] - j # x
            box[2] = self.S * box[2] - i # y
            box[3] = self.S * box[3] # w
            box[4] = self.S * box[4] # h

            if label_matrix[i,j,20] == 0:
                # one-hot vector for class
                label_matrix[i, j, box_class] = 1

                # confidence for ground-truth = 1
                label_matrix[i, j, 20] = 1

                # box coordinate
                label_matrix[i, j, 21:25] = torch.tensor(box[1:])


        return image, label_matrix
