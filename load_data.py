import os
import numpy as np
import PIL.Image as Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt


def load_train_test_splits(path_to_imgdir,test_size=0.1,val_size=0.1,seed=0,normalize_y2=True,augment=False):
    """ 
    Retruns train, test (and val) splits of X, Y1, Y2 contained in path 
    path contains folders named 'region_index,coordinates'
    """


    paths = [x[0] for x in os.walk(path_to_imgdir)]
    paths.remove('images')

    X = []
    Y1 = []
    Y2 =[]
    count = 0
    for path in tqdm(paths,desc='Loading and Processing Data',ascii=" ▖▘▝▗▚▞█"):
    # for path in tqdm(np.random.choice(paths,1000),desc='Loading and Processing Data',ascii=" ▖▘▝▗▚▞█"):
        imgs_paths = os.listdir(path)
        try:
            imgs_paths = [i.decode('UTF-8') for i in imgs_paths]
        except:
            pass
        Y = [int(x.split('.')[0]) for x in imgs_paths]
        imgs_paths = [x for _, x in sorted(zip(Y, imgs_paths), key=lambda pair: pair[0])]
        arrs = []
        for img_path in imgs_paths:
            img_path = path + '/' + img_path
            image_arr = np.asarray(Image.open(img_path))
            arrs.append(image_arr)
        thisX = np.concatenate(tuple(arrs),axis=1)
        y1 = np.zeros(88)
        y1[int(path.split('/')[1].split(',')[0])] = 1
        y2 = path.split('/')[1].split(',')[1:]
        y2 = [float(i) for i in y2]
        y1 = np.array(y1)
        y2 = np.array(y2)

        X.append(thisX)
        Y1.append(y1)
        Y2.append(y2)


    X = np.stack(X)
    Y1 = np.stack(Y1)
    Y2 = np.stack(Y2)

    if normalize_y2:
        long_norm = np.linalg.norm(Y2[:,0])
        Y2[:,0] = Y2[:,0]/long_norm

        lat_norm = np.linalg.norm(Y2[:,1])
        Y2[:,1] = Y2[:,1]/lat_norm

    if augment:
        augmented_X = []
        augmented_Y1 = []
        augmented_Y2 =[]
        def augment(image):
            new_im = np.flip(image,axis=1)
            return new_im

        for idx,image in enumerate(tqdm(X,desc='Adding Augmented Data',ascii=" ▖▘▝▗▚▞█")):
            new_image = augment(image)
            augmented_X.append(new_image)
            augmented_Y1.append(Y1[idx])
            augmented_Y2.append(Y2[idx])

        augmented_X = np.stack(augmented_X)
        augmented_Y1 = np.stack(augmented_Y1)
        augmented_Y2 = np.stack(augmented_Y2)

        X = np.concatenate((X,augmented_X),axis=0)
        Y1 = np.concatenate((Y1,augmented_Y1),axis=0)
        Y2 = np.concatenate((Y2,augmented_Y2),axis=0)

    X_train, X_test, y1_train, y1_test,y2_train, y2_test= train_test_split(X,Y1,Y2,stratify=Y1,test_size=test_size,random_state=seed)

    X_train, X_val, y1_train, y1_val,y2_train, y2_val= train_test_split(X_train, y1_train,y2_train, stratify=y1_train, test_size=val_size/(1-test_size), random_state=seed)  # 0.11 * 0.9 ≈ 0.1


    print('Train size: ', X_train.shape[0])
    print('Test size: ', X_test.shape[0])
    print('Val size: ', X_val.shape[0])

    print('X shape: {},    y1 shape: {},    y2 shape: {}'.format(X_val.shape[1:],y1_val.shape[1:],y2_val.shape[1:]))


    if normalize_y2:
        return X_train,y1_train,y2_train,X_test,y1_test,y2_test,X_val,y1_val,y2_val,long_norm,lat_norm
    else:
        return  X_train,y1_train,y2_train,X_test,y1_test,y2_test,X_val,y1_val,y2_val

