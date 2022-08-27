import os
import numpy as np
import cv2

def load():
    """
    This function rerurns a 'data' containing all images paths, 'target' containing the image labels
    for each eye.
    """
    data = []
    target = []
    target_i = 0
    index_used = False
    for dirpath, dirnames, filenames in os.walk('C:\\Users\\nikhi\\Downloads\\UTIRIS V.1\\Infrared Images'):
        for f in filenames:
            if f.endswith('.bmp'):
                data.append('{}\{}'.format(dirpath, f))
                target.append(target_i)
                index_used = True
        if index_used:
            target_i += 1
            index_used = False
    return {'data': np.array(data),
            'target': np.array(target)}

if __name__ == '__main__':
    
    data = load()['data']
    for i in data:
        print(i)
        image = cv2.imread(i)
        cv2.imshow('test', image)
        cv2.waitKey(0)
