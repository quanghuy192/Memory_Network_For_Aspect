# -*- coding: utf-8 -*-

from underthesea import word_tokenize
from underthesea import pos_tag
import csv
try:
    import cPickle as pickle
except:
    import pickle

# vn_sentences = word_tokenize('Do dịch bệnh, nên sinh viên được nghỉ thêm một tuần', format='text')
# vn_sentences_pos_tag = pos_tag(vn_sentences)
# [('Do', 'E'), ('dịch_bệnh', 'N'), (',', 'CH'), ('nên', 'C'), ('sinh_viên', 'V'), ('được', 'R'), ('nghỉ', 'V'), ('thêm', 'V'), ('một', 'M'), ('tuần', 'N')]
#
# print(vn_sentences)
# print(vn_sentences_pos_tag)

with open('foody_data.pkl', 'rb') as fp:
    data_frame_foody_data = pickle.load(fp)
    print(type(data_frame_foody_data))
    # print(data_frame_foody_data)
    # for col in data_frame_foody_data.columns:
    #     print(col)

    data_frame_foody_data_filtered = data_frame_foody_data[['review_content', 'avg_score', 'location_point', 'space_point', 'quality_point',
                                                            'service_point', 'price_point']]

    data_frame_foody_data_filtered1 = data_frame_foody_data[
        ['review_content']]

    data_frame_foody_data_filtered2 = data_frame_foody_data[
        ['location_point', 'space_point', 'quality_point',
         'service_point', 'price_point']]

    print(data_frame_foody_data_filtered1)

    print(data_frame_foody_data_filtered2)

    f = open("data.txt", "w+")
    for index, row in data_frame_foody_data_filtered1.iterrows():
        print(row['review_content'])
        f.writelines(row['review_content'])
    f.close()

