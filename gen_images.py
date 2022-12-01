import google_streetview.api
import shutil
import numpy as np
import json
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import os
from os import walk
def gen_image(bounds,name):

    ymin,xmin,ymax,xmax = bounds
    cord = [np.random.uniform(xmin,xmax),np.random.uniform(ymin,ymax)]
    heading = np.random.randint(0,120)
    params = [{
        'size': '400x400', # max 400x400 pixels
        'location': '{},{}'.format(cord[1],cord[0]), # coordinates
        'heading': str(heading), # the rotation angle around the camera locus in degrees relative from true north
        'pitch': str(0), # specifies the up or down angle of the camera relative to the Street View vehicle
        'key': 'insert key here'
        }]
    results = google_streetview.api.results(params)
    results.download_links('images')
    with open('images/metadata.json') as json_file:
        data = json.load(json_file)
        status = data[0]['status']
    if status == 'OK':
        location_path = 'images/{},{},{}'.format(name,cord[1],cord[0])
        os.mkdir(location_path)
        shutil.copy('images/gsv_0.jpg', location_path+'/{}.jpg'.format(heading))
        for z in range(1,3):
            heading = np.random.randint(120*z,(120*z)+120)
            params[0]['heading'] = heading
            results = google_streetview.api.results(params)
            results.download_links('images')
            shutil.copy('images/gsv_0.jpg', location_path+'/{}.jpg'.format(heading))
        return True
    else:
        return False


def main():
    polygrid = pickle.load(open("PolyGrid.pkl",'rb'))
    print('Total Squares : ', len(polygrid.keys()))
    for square_id in range(len(polygrid.keys())): ## 88 squares in the grid
        if square_id in [73,74]:
            print('Square ', square_id)
            poly = Polygon((polygrid[square_id]))
            count = 5 if square_id == 73 else 0 
            failed = 0
            pbar = tqdm(total = 200-count)
            while count < 200:
                if gen_image(poly.bounds,square_id): # if found image
                    pbar.update(1)
                    count += 1
                    failed = 0
                else:
                    failed += 1
                    if failed > 10000:
                        print('square id {} failed after it found {} images'.format(square_id,count))
                        break 

    
if __name__ == '__main__':
    main()
 
