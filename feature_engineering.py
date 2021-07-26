from data_classes import FDDBAnnotation
import cv2
import numpy as np
import os
from os import path
from PIL import Image
import io
from object_detection.utils import dataset_util, label_map_util
import tensorflow as tf
if __package__:
    from .data_classes import FDDBExample
else:
    from data_classes import FDDBExample

label_map = label_map_util.load_labelmap(r"training/label_map.pbtext")
label_map_dict = label_map_util.get_label_map_dict(label_map)

def class_text_to_int(row_label):
    return label_map_dict[row_label]

def read_folds(fold_start=1, fold_end=1):
    for i in range(fold_start,fold_end+1):
        fpath = path.join('labels', f'FDDB-fold-{str(i).zfill(2)}-ellipseList.txt')
        with open(fpath, 'r') as f:
            img_path = None
            n_faces = None
            annotations = []
            for line in f:
                line = line.strip()
                if not img_path:
                    img_path = line+'.jpg'
                elif not n_faces:
                    n_faces = int(line)
                else:
                    #bounding box
                    n_faces -=1
                    ellipse_def = line[:line.index('  ')].split(' ')
                    ellipse_def = [float(item) for item in ellipse_def]
                    annotaton= FDDBAnnotation(
                        major_axis_radius= ellipse_def[0],\
                        minor_axis_radius= ellipse_def[1],\
                        angle=ellipse_def[2],\
                        center_x=ellipse_def[3],\
                        center_y= ellipse_def[4])
                    annotations.append(annotaton)
                    if n_faces==0:
                        #reset
                        yield FDDBExample(fold= i,\
                            image_path= img_path,\
                            annotations= annotations\
                        )
                        img_path, n_faces, annotations = None, None, []
                        
def create_tf_example(example:FDDBExample)-> tf.train.Example:
    with tf.io.gfile.GFile(os.path.join('images', example.image_path), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = path.basename(example.image_path.encode('utf8'))
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for annotation in example.annotations:
        (x1, y1), (x2, y2) = annotation.cv2_rectangle_box()
        xmins.append(x1 / width)
        xmaxs.append(x2 / width)
        ymins.append(y1 / height)
        ymaxs.append(y2 / height)
        classes_text.append('face'.encode('utf-8'))
        classes.append(class_text_to_int('face'))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example
                        
def generate_tf_records(output_path, fold_start=1, fold_end=1):
    print('Generating tf-record at:', output_path)
    sample_count = 0
    writer = tf.io.TFRecordWriter(output_path)
    for example in read_folds(fold_start, fold_end):
        tf_example = create_tf_example(example)
        sample_count += 1
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('sample found:', sample_count)

if __name__ == '__main__':
    
    def test1():
        preview_index = 4500
        i = 0
        for item in read_folds(1,10):
            i += 1
            if i == preview_index:
                break
        impath = path.join('images', item.image_path)
        image = cv2.imread(impath)
        #item.angle, 0, 360, 'red', 5
        for annotation in item.annotations:
            ellipse_box = annotation.cv2_ellipse_box()
            cv2.ellipse(image, ellipse_box, (0, 204, 0), 3)
            p1, p2 = annotation.cv2_rectangle_box()
            cv2.rectangle(image, p1, p2, (0, 0, 204), 3)
        cv2.imshow("preview", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    generate_tf_records(r'training/train.record', fold_start=1, fold_end=8)
    generate_tf_records(r'training/test.record', fold_start=9, fold_end=10)