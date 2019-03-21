# -*- coding: UTF-8 -*-

import os
import sys
import xml.etree.ElementTree as ET

class Data_prep(object):

    def __init__(self, root_path, pic_file_path, txt_file_path, xml_file_path, file_index):

        # File directory (pic, xml, txt)
        # pic_file_path should be stored in txt file for training.
        self.root_path  = root_path
        self.pic_file_path = pic_file_path
        self.txt_file_path_pos, self.txt_file_path_neg = txt_file_path
        self.xml_file_path = xml_file_path

        # Current file index
        self.file_index = file_index

        # initial variables for bndbox information for each image.
        self.bndbox_str = ""
        self.num_bndbox = 0


    def read_xml_file(self):
        """
        Read the xml file and return the root
        
        Output:
            - root, the root of xml tree
        """

        #Find the root of xml tree.
        xml_tree = ET.parse(self.xml_file_path + "pic{}.xml".format(self.file_index))
        root = xml_tree.getroot()

        return root

    def get_bndbox(self, root):
        """
        Collect the information required by training specification.
        Input: 
            - xml tree root
        Output: 
            - bndbox_dic, a dic that contains the bounding box information
            - num_bndbox, the number of bndbox in this pic
        
        """
        # bndbox_dic = {}

        # tag = ET.Element.tag
        # attrib = ET.Element.attrib
        # Value = ET.Element.text

        # print(root[6][4][0].text)
        for obj in root.iter("object"):
            self.num_bndbox += 1
            for bndbox in obj.iter("bndbox"):
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)
                x, y, w, h = self.PascalVOC_to_txt(xmin, ymin, xmax, ymax)
                self.bndbox_str += "{} {} {} {} ".format(x, y, w, h)
                # bndbox_dic["obj%s"%num_bndbox] = PascalVOC_to_txt(xmin, ymin, xmax, ymax)
        return self.bndbox_str, self.num_bndbox

    def PascalVOC_to_txt(self, xmin, ymin, xmax, ymax):
        """
        Tansform Pascal bounding box to training input format

        """
        x = xmin
        y = ymin
        w = abs(xmax - xmin)
        h = abs(ymax - ymin)
        return (x, y, w, h)


    def write_text_file(self):
        """
        This function is intended to writing a .txt file
        for model training.
        (format: filename n x y w h x y w h)
        
        """

        # txt_str = self.make_txt_str()
        if self.num_bndbox:
            f = open(self.txt_file_path_pos, "a+")
            pic_path = self.pic_file_path + "\pic{}.jpg".format(self.file_index)
            txt_str = "{} {} {}\n".format(pic_path, self.num_bndbox, self.bndbox_str)
        else:
            f = open(self.txt_file_path_neg, "a+")
            pic_path = self.pic_file_path + "\pic{}.jpg".format(self.file_index)
            txt_str = "{}\n".format(pic_path)

        f.write(txt_str)
        f.close()
        # print(txt_str)
    
    def generate(self):
        root = self.read_xml_file()
        self.get_bndbox(root)
        self.write_text_file()
        #print(root_path)

# Get the current .py file root path.
root_path = os.path.dirname(os.path.realpath(__file__))

pic_file_path = root_path + "\pic"
txt_file_path_pos = root_path + "/training_input_txt/training_list_pos.txt"
txt_file_path_neg = root_path + "/training_input_txt/training_list_neg.txt"
xml_file_path = root_path + "/xml/"
txt_file_path = (txt_file_path_pos, txt_file_path_neg)

# Scan all the xml and generate string one by one.
file_total = 0
for file in os.listdir(pic_file_path):
    sub_path = os.path.join(pic_file_path, file)
    if os.path.isfile(sub_path):
        file_total += 1

        # Generate txt file to record the data info.
        gl = Data_prep(root_path, pic_file_path, txt_file_path, xml_file_path, file_total)
        gl.generate()
