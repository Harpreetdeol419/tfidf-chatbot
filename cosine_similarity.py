from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def GetVec(texts): # words -> vectors
	return TfidfVectorizer(stop_words='english').fit_transform(texts).toarray()
	
def NumSim(VecOne, VecSec): # compute cosine similarity between two vectors
	ArrOne = np.array(VecOne).flatten()
	ArrSec = np.array(VecSec).flatten()
	if ArrOne.size == 0 or ArrSec.size == 0:
		return 0
	MagOne = np.linalg.norm(ArrOne)
	MagSec = np.linalg.norm(ArrSec)
	if MagOne == 0 or MagSec == 0:
		return 0
	DotPro = np.dot(ArrOne, ArrSec)
	result = DotPro / (MagOne * MagSec)
	return  result
	
def test(): # inn
	vocab = [ # kept two sentences cuz NumSim accepts two parameters only
	"hi bro",
	"bye bro",
	]
	Vecs = GetVec(vocab)
	VecOne = Vecs[0]
	VecSec = Vecs[1]
	sim = NumSim(VecOne, VecSec)
	print(sim) # print similarity
test()
