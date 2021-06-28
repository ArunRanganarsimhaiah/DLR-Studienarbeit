import glob
import ntpath
import os
import pandas as pd
import tensorflow as tf
import xml.etree.ElementTree as ET

#from tensorflow.train import BytesList, Feature, Features, Example


# function encode all .png files in source (including subfolders) and write them to .tfrecord file
def write_tfrecord(source, target):
    with tf.io.TFRecordWriter(target) as f:
        for filename in glob.glob(os.path.join(source, '*.png')):
            img_binary = tf.io.read_file(filename).numpy()
            filename_binary = bytes(ntpath.basename(filename), encoding='utf-8')
            #img_example = Example(
              #  features=Features(
               #     feature={'image': Feature(bytes_list=BytesList(value=[img_binary])),
                  #           'filename': Feature(bytes_list=BytesList(value=[filename_binary]))}))
            #f.write(img_example.SerializeToString())


# function to decode serialized examples in .tfrecord dataset
def preprocess(example):
    features = {'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
                'filename': tf.io.FixedLenFeature([], tf.string, default_value='')}
    parsed_example = tf.io.parse_single_example(example, features)
    image = tf.io.decode_png(parsed_example['image'], channels=3, dtype=tf.uint8)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, parsed_example['filename']


categories = {'none': 0, 'wrinkle': 1, 'twist': 2, 'foreign material': 3, 'overlap': 4, 'gap': 5}


# function to write xml information to pandas dataframe
def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    return pd.DataFrame(xml_list, columns=column_name)
