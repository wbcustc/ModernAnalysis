
data = []
temp = []
meaning = []

with open('votesmart_20bill.txt') as file1:
	last_tag = -1	
	for line in file1:
		line = line.strip()
		index = line.find('I')
		if index != -1:
			tag = int(line[index+1:])
			if tag <= last_tag:
				if len(temp) > 0:
					data.append(temp)
				temp = []
				temp.append(tag)
			else:
				temp.append(tag)
			last_tag = tag

with open('votesmart_20bill_meaning.txt') as file2:
	for line in file2:
		line = line.strip()
		index = line.find('S')
		if index != -1:
			meaning.append(line[index+2:-1])


# From Harrington 11.5
def createC1(dataSet):
	C1 = []
	for transaction in dataSet:
		for item in transaction:
			if not [item] in C1:
				C1.append([item])
	C1.sort()
	return map(frozenset, C1)

def scanD(D, Ck, minSupport):
	ssCnt = {}
	for tid in D:
		for can in Ck:
			if can.issubset(tid):
				if not ssCnt.has_key(can): 
					ssCnt[can]=1
				else: 
					ssCnt[can] += 1
	numItems = float(len(D))
	retList = []
	supportData = {}
	for key in ssCnt:
		support = ssCnt[key]/numItems
		if support >= minSupport:
			retList.insert(0,key)
		supportData[key] = support
	return retList, supportData

def aprioriGen(Lk, k): #creates Ck
	retList = []
	lenLk = len(Lk)
	for i in range(lenLk):
		for j in range(i+1, lenLk):
			L1 = list(Lk[i])[:k-2]; 
			L2 = list(Lk[j])[:k-2]
			L1.sort(); 
			L2.sort()
			if L1==L2:
				retList.append(Lk[i] | Lk[j])
	return retList

def apriori(dataSet, minSupport = 0.5):
	C1 = createC1(dataSet)
	D = map(set, dataSet)
	L1, supportData = scanD(D, C1, minSupport)
	L = [L1]
	k = 2
	while (len(L[k-2]) > 0):
		Ck = aprioriGen(L[k-2], k)
		Lk, supK = scanD(D, Ck, minSupport)
		supportData.update(supK)
		L.append(Lk)
		k += 1
	return L, supportData

def getFrequentItemset():
	minSupport = 0.5
	while minSupport >= 0.3:
		L,support = apriori(data,minSupport)
		ret = str(len(L))
		for s in L:
			ret += ' ' + str(len(s))
		print ret
		minSupport -= 0.05

def getKey(item):
	return -item[2]

def generateRules(L, supportData, minConf=0.7):
	bigRuleList = []
	for i in range(1, len(L)):
		for freqSet in L[i]:
			H1 = [frozenset([item]) for item in freqSet]
			if (i > 1):
				rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
			else:
				calcConf(freqSet, H1, supportData, bigRuleList, minConf)
	return sorted(bigRuleList,key=getKey)

def calcConf(freqSet, H, supportData, brl, minConf=0.7):
	prunedH = []
	for conseq in H:
		conf = supportData[freqSet]/supportData[freqSet-conseq]
		if conf >= minConf:
			# print freqSet-conseq,'-->',conseq,'conf:',conf
			brl.append((freqSet-conseq, conseq, conf))
			prunedH.append(conseq)
	return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
	m = len(H[0])
	if (len(freqSet) > (m + 1)):
		Hmp1 = aprioriGen(H, m + 1)
		Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
		if (len(Hmp1) > 1):
			rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

def displayRule(bigRuleList):
	for rule in bigRuleList:
		print rule[0],'-->',rule[1],'conf:',rule[2]

def main():
	getFrequentItemset()
	L,supp = apriori(data,0.35)
	rules = generateRules(L, supp,0.99)
	displayRule(rules)
	return

if __name__ == '__main__':
	main()

			