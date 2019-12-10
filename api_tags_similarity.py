import re
import numpy as np

def save_tag_to_dic(tags, tags_list_dic, dic, pt):
    for i in range(0, len(tags)):
        if tags[i] != pt:  ## ingore the positive tag
            tags[i] = tags[i][1:-1]
            if tags[i] not in dic:
                tags_list_dic[tags[i]] = 1
                dic[tags[i]] = 1
            else:
                dic[tags[i]] += 1
    return tags_list_dic, dic

def cal_sim(tags_list_dic, dic1, n1, dic2, n2):
    sps = []
    sns = []
    for key in tags_list_dic:
        if key in dic1:
            sps.append(dic1[key]/n1)
        else:
            sps.append(0)
        if key in dic2:
            sns.append(dic2[key]/n2)
        else:
            sns.append(0)
    return np.dot(sps, sns) / (np.linalg.norm(sps) * np.linalg.norm(sns))

def api_tag_sim(df, ptag):
    tags_list_dic = {}
    n_p = 0
    n_n = 0
    pts_dic = {}
    nts_dic = {}
    
    for i in range(0, df.shape[0]):
        tags = df.iloc[i].Tags
        tags = re.findall('<.*?>', tags, re.S)
        if ptag in tags:
            n_p += 1
            tags_list_dic, pts_dic = save_tag_to_dic(tags, tags_list_dic, pts_dic, ptag)
        else:
            n_n += 1
            tags_list_dic, nts_dic = save_tag_to_dic(tags, tags_list_dic, nts_dic, ptag)
    return cal_sim(tags_list_dic, pts_dic, n_p, nts_dic, n_n)

def api_tag_sim_pp(df, ptag):
    tags_list_dic = {}
    n_p1 = 0
    n_p2 = 0
    n_p = 0
    p1ts_dic = {}
    p2ts_dic = {}
    
    for i in range(0, df.shape[0]):
        tags = df.iloc[i].Tags
        tags = re.findall('<.*?>', tags, re.S)
        if ptag in tags:
            n_p += 1
            if n_p%2 == 0:
            	n_p1 += 1
            	tags_list_dic, p1ts_dic = save_tag_to_dic(tags, tags_list_dic, p1ts_dic, ptag)
            else:
            	n_p2 += 1
            	tags_list_dic, p2ts_dic = save_tag_to_dic(tags, tags_list_dic, p2ts_dic, ptag)
    return cal_sim(tags_list_dic, p1ts_dic, n_p1, p2ts_dic, n_p2)

def cal_list_sim(list1, list2):
    tags_list_dic = {}
    n_1 = len(list1)
    n_2 = len(list2)
    pts_dic = {}
    nts_dic = {}
    
    for i in range(n_1):
    	words = re.split(' ', list1[i])
    	for word in words:
    		if word in pts_dic:
    			pts_dic[word] += 1
    		else:
    			tags_list_dic[word] = 1
    			pts_dic[word] = 1

    for i in range(n_2):
    	words = re.split(' ', list2[i])
    	for word in words:
    		if word in nts_dic:
    			nts_dic[word] += 1
    		else:
    			tags_list_dic[word] = 1
    			nts_dic[word] = 1
    return cal_sim(tags_list_dic, pts_dic, n_1, nts_dic, n_2)
