from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def GetVec(text):
	return TfidfVectorizer(stop_words="english").fit_transform(text).toarray()
	
def NumSim(VecOne, VecSec):
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
	return result
	
def run(): # just for example you can add your questions and answers
    que = [
    "how to reset password",
    "how to contact support"
    ]
    ans = [
    "Click reset password on login page",
    "Email us at xyz@example.com"
    ]
    while True: # run again and again
	    user = input("User: ")
	    if user:
	    	combo = [user] + que
	    	all = GetVec(combo)
	    	usr = all[0]
	    	qse = all[1:]
	    	scores = [NumSim(usr, q) for q in qse]
	    	BestIndex = np.argmax(scores)
	    	BestScore = scores[BestIndex]
	    	if BestScore < 0.2:
	    		print("I dont know yet")
	    	else:
	    		print("AI:", ans[BestIndex])
run()
