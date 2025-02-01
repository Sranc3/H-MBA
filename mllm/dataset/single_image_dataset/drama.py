import os
import numpy as np
import glob
# import pickle as pkl #memory error for huge pkl files
import dill as pkl

import torch
from torch.utils import data
from tqdm import tqdm
import string
import cv2
from torch.nn.utils.rnn import pad_sequence
import json

def my_collate(batch):
# img, enc_caption, caption_len, bbox, dims, caption, img_path
		img = [item[0] for item in batch]
		enc_caption = [item[1] for item in batch]
		caption_len = [item[2] for item in batch]
		bbox = [item[3] for item in batch]
		dims = [item[4] for item in batch]
		# caption = [item[5] for item in batch]
		# img_path = [item[6] for item in batch]

		img = torch.from_numpy(np.stack(img))
		enc_caption = torch.from_numpy(np.stack(enc_caption))
		caption_len = torch.from_numpy(np.stack(caption_len))
		bbox = torch.from_numpy(np.stack(bbox))
		dims = torch.from_numpy(np.stack(dims))

		return [img, enc_caption, caption_len, bbox, dims]

def to_categorical(y, num_classes):
				""" 1-hot encodes a tensor """
				return np.eye(num_classes, dtype='uint8')[y]

class DramaDataset(data.Dataset):
		def __init__(self, args, phase):
				'''
				Drama dataset object.
				Params:
						args: arguments passed from main file
						phase: 'train' or 'val'
				'''
				self.args = args
				result_file = self.args.data_root+"/integrated_output_v2.json"
				with open(result_file, 'r') as j:
					self.in_data = json.loads(j.read())
				self.out_data = []
				self.out_data2 = []
				self.word_map = {}
				self.max_length = 0
				table = str.maketrans('', '', string.punctuation)
				if self.args.preprocess and phase=="train": ## pre-processing val and test in train itself
						print("preparing ",phase," data")
						for result in tqdm(self.in_data):
								data = {}
								# url = result[0]
								url = result["s3_fileUrl"]
								risk = result["Risk"]
								if risk=="No":
									continue
								caption = result["Caption"]
								if caption=="N/A":
									continue
								bbox = result["geometry"]
								if len(bbox)==0:
									continue

								img_path = url.replace("https://s3-us-west-2.amazonaws.com/hrius.scaleapi/data/drama",args.data_root)
								img = cv2.imread(img_path)
								flow_img_path = img_path.replace("frame_","flow_")
								flow_img = cv2.imread(flow_img_path)
								if not (os.path.exists(img_path) and os.path.exists(flow_img_path)): #if file exists
									continue
								img_h = np.shape(img)[0]
								img_w = np.shape(img)[1]
								# caption = result[1]
								x = np.clip(int(float(bbox[0][0])), 0, img_w)/img_w
								x1 = np.clip(int(float(bbox[2][0])), 0, img_w)/img_w
								y = np.clip(int(float(bbox[0][1])), 0, img_h)/img_h
								y1 = np.clip(int(float(bbox[2][1])), 0, img_h)/img_h
								# print(x,x1,y,y1)
								width = abs(x1-x)
								height = abs(y1-y)
								# print(w, h)
								assert (x<=1 and x>=0) and (y<=1 and y>=0) and (width<=1 and width>=0) and (height<=1 and height>=0) # normalized and within the limits
								img = cv2.resize(img,(1000, 740))
								# print(img_path.replace("frame_","flow_"))
								try:
										flow_img = cv2.resize(flow_img,(1000, 740))
								except:
										continue

								data["img"] = np.transpose(img, (2, 0, 1))#c, w ,h
								data["flow_img"] = np.transpose(flow_img, (2, 0, 1))#c, w ,h
								data["img_path"] = img_path


								### caption cleaning
								# tokenize
								caption = caption.split()
								# convert to lower case
								caption = [word.lower() for word in caption]
								# remove punctuation from each token
								caption = [w.translate(table) for w in caption]
								# remove hanging 's' and 'a'
								caption = [word for word in caption if len(word)>1]
								# remove tokens with numbers in them
								caption = [word for word in caption if word.isalpha()]

								# including start and end
								if len(caption)+2>self.max_length:
										self.max_length=len(caption)+2
								data["caption_len"] = [len(caption)+2]
								# store as string
								data["caption"] = '<startseq> ' +' '.join(caption)+ ' <endseq>'


								data["bbox"] = [x,y,width,height]
								data["dims"] = [1000, 740]
								self.out_data.append(data)

						#create word map, word2ix and ix2word
						self.vocabulary = set()
						for i in range(len(self.out_data)):
								self.vocabulary.update(self.out_data[i]["caption"].split())
						print('Original Vocabulary Size: %d' % len(self.vocabulary))
						self.ixtoword = {}
						self.wordtoix = {}
						self.wordtoix["<pad>"] = 0
						self.ixtoword[0] = "<pad>"
						ix = 1 # 0 is for padding
						for w in self.vocabulary:
								self.wordtoix[w] = ix
								self.ixtoword[ix] = w
								ix += 1


						with open(self.args.data_root+'/wordtoix.pkl', 'wb') as fp:
								pkl.dump(self.wordtoix, fp)
						with open(self.args.data_root+'/ixtoword.pkl', 'wb') as fp:
								pkl.dump(self.ixtoword, fp)

						## encode the captions using vocabulary and save caption lengths
						for i in range(len(self.out_data)):
								desc = self.out_data[i]["caption"]
								# encode the sequence
								seq = [self.wordtoix[word] for word in desc.split(' ') if word in self.wordtoix]
								seq_w_padding = torch.zeros(self.max_length, dtype=torch.int64)
								seq_w_padding[0:len(seq)] = torch.LongTensor(seq)
								self.out_data[i]["enc_caption"] = seq_w_padding
								# split one sequence into multiple X, y pairs
								# for i in range(1, len(seq)):
								#   data={}
								#   # split into input and output pair
								#   in_seq, out_seq = seq[:i], seq[i]
								#   # pad input sequence
								#   # encode output sequence
								#   out_seq = to_categorical([out_seq], num_classes=len(self.vocabulary)+1)[0]
								#   # store
								#   data["img"] = self.out_data[i]["img"]
								#   data["bbox"] = self.out_data[i]["bbox"]
								#   data["dims"] = self.out_data[i]["dims"]
								#   data["caption"] = in_seq2
								#   data["out_seq"] = out_seq
								#   if i==len(seq)-1:
								#       data["full_sequence"] = [1]
								#   else:
								#       data["full_sequence"] = [0]
								#   self.out_data2.append(data)

						## save train
						out_datas = self.out_data[0:int(0.7*len(self.out_data))]
						print("saving train data..")
						for ind, out_data in enumerate(tqdm(out_datas)):
								out_file = self.args.data_root+"/processed/train/"+str(ind).zfill(6)+".pkl"
								pkl.dump(out_data, open(out_file,'wb'))

						## save val
						print("saving val data..")
						out_datas = self.out_data[int(0.7*len(self.out_data)):int(0.85*len(self.out_data))]
						for ind, out_data in enumerate(tqdm(out_datas)):
								out_file = self.args.data_root+"/processed/val/"+str(ind).zfill(6)+".pkl"
								pkl.dump(out_data, open(out_file,'wb'))

						## save test
						print("saving test data..")
						out_datas = self.out_data[int(0.85*len(self.out_data)):len(self.out_data)]
						for ind, out_data in enumerate(tqdm(out_datas)):
								out_file = self.args.data_root+"/processed/test/"+str(ind).zfill(6)+".pkl"
								pkl.dump(out_data, open(out_file,'wb'))

				else:
						# self.out_data = pkl.load(open(self.out_file, 'rb'))
						self.ixtoword = pkl.load(open(self.args.data_root+'/ixtoword.pkl','rb'))
						self.wordtoix = pkl.load(open(self.args.data_root+'/wordtoix.pkl','rb'))

				self.out_files = glob.glob(self.args.data_root+"/processed/"+phase+"/*.pkl")

		def __len__(self):
				return len(self.out_files)

		def __getitem__(self, index):
				result = pkl.load(open(self.out_files[index], 'rb'))
				# image = result["img"].transpose(1,2,0)
				# cv2.imshow("image",image)
				# cv2.waitKey(10000)
				img = torch.FloatTensor(result["img"]).to(self.args.device)
				flow_img = torch.FloatTensor(result["flow_img"]).to(self.args.device)
				enc_caption = torch.LongTensor(result["enc_caption"]).to(self.args.device)
				caption_len = torch.LongTensor(result["caption_len"]).to(self.args.device)
				bbox = torch.FloatTensor(result["bbox"]).to(self.args.device)
				dims = torch.FloatTensor(result["dims"]).to(self.args.device)
				# caption = result["caption"]
				# img_path = result["img_path"]
				# print(torch.min(img),torch.min(flow_img),torch.min(enc_caption),torch.min(caption_len),torch.min(bbox),torch.min(dims))
				# print(torch.max(img),torch.max(flow_img),torch.max(enc_caption),torch.max(caption_len),torch.max(bbox),torch.max(dims))
				return img, flow_img, enc_caption, caption_len, bbox, dims