import qutip
import scipy
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import math

import scipy.linalg

Z = qutip.sigmaz()
X = qutip.sigmax()
Y = qutip.sigmay()
I = qutip.identity(dimensions=2)
# though it does define n_j = 0.5(I - Z_j)
n_hat = 0.5 * (I - Z) # page 11 of paper

# imported this from old code, and am just assuming that it works correctly
# defines an operator on a certain qubit within a certain
# system size
# num_qubits > 0
# act_on_qubit int in [1, num_qubits]
# qubits are 1-indexed here
def sub_i(operator, act_on_qubit:int, num_total_qubits:int) -> qutip.Qobj:
	act_on_qubit -= 1 # change to be computer science index
	if num_total_qubits <= 0: 
		raise TypeError("num_qubits invalid")
	if act_on_qubit < 0 or act_on_qubit > num_total_qubits - 1: 
		raise TypeError("acting qubit out of range")

	result = None # just to define this
	for i in range(num_total_qubits):
		if i == 0:
			if i == act_on_qubit:
				result = operator	
			else:
				result = I
		elif i == act_on_qubit:
			result = qutip.tensor(result, operator)
		else:
			result = qutip.tensor(result, I)
	return qutip.Qobj(result)

# this function was written to hide some test code
def checkSubI_with_Nhat():
	## checking the nhat operator with the sub_i function
	## so it doesnt look like there are any differences in the way I use the n_hat operator
	for i in range(1, 5):
		nHatOrig = 0.5 * (I - Z)
		v1 = sub_i(nHatOrig, i, 4)
		v2 = (0.5 * sub_i(I, i, 4) 
			- 0.5 * sub_i(Z, i, 4))

		# print(f"v1:{v1}")
		# print(f"v2:{v2}")

		print(f"difference:{v1 == v2}")

	## the get bitstrings 1d function from the paper's github
	## I'm not entirely sure what its doing, but I need it to create the projection operators
	return None

def get_bitstrings_1d(N, encoding):
    bitstrings = []

    if encoding == "unary":
        bitstring = (N-1) * ["0"]
        for i in range(N):
            bitstrings.append("".join(bitstring))
            if i < N - 1:
                bitstring[i] = "1"

        return bitstrings
    
    elif encoding == "antiferromagnetic":
        bitstring = []
        for i in range(N-1):
            if i % 2 == 0:
                bitstring.append("0")
            else:
                bitstring.append("1")

        for i in range(N):
            bitstrings.append("".join(bitstring))
            if i < N - 1:
                if i % 2 == 0:
                    bitstring[i] = "1"
                else:
                    bitstring[i] = "0"

        return bitstrings
        
    elif encoding == "one-hot":
        bitstring = N * ["0"]
        for i in range(N):
            bitstring[i] = "1"
            if i > 0:
                bitstring[i-1] = "0"

            bitstrings.append("".join(bitstring))

        return bitstrings
    else:
        return ValueError("Encoding not supported.")

# wrapper for matrix multiplication so I can use it in functional calls
def mat_mult(A, B):
	return A @ B

def frobeniusNorm(M:np.ndarray):
	return np.linalg.norm(M, ord='fro')	
	## old implementation
	# def frobeniusNorm(A:np.ndarray):
	# 	return np.sqrt(np.trace(A.conj() @ A))

# wrapper for the numpy spectral norm function
def spectralNorm(M:np.ndarray):
	return np.linalg.norm(M, ord=2)

# general class defintion for the embeddings
class HamiltonianEmbedding:
	def __init__(self, A:np.ndarray):
		(self.dim, x) = A.shape
		if self.dim != x:
			raise TypeError("A Array must be square")
		self.A = A

	def HPen(self) -> np.ndarray:
		raise NotImplementedError("subclasses must implement this method")

	def Q_a(self) -> np.ndarray:
		raise NotImplementedError("subclasses must implement this method")

	# note that the type annotation of float accepts int
	# according to PEP 484 from 2018
	def H_embedding(self, g:float) -> np.ndarray:
		return g * self.HPen() + self.Q_a()

	def restrictToS(self, M):
		raise NotImplementedError("subclasses must implement this method")

	def deprecatedProjHider():
		## deprecated by me -- dont use this	
		# def Proj(self) -> np.ndarray:
		# 	raise NotImplementedError("subclasses must implement this method")

		## deprecated by me -- dont use this
		# # for a specific original nxn matrix A (defined by the unary embedding instance)
		# # and a given 2^(n-1) x 2^(n-1) matrix M
		# # it projects M to nxn and then actually reduces the dimension of the underlying structure
		# def ProjRedDim(self, M:np.ndarray) -> np.ndarray:
		# 	assert isinstance(M, np.ndarray) # Qobj has weird dimensions

		# 	P = self.ProjM()
		# 	if isinstance(P, qutip.Qobj): # it has weird dimesions, so we fix that
		# 		P = P.full() # turns it into a np.ndarray

		# 	Mprime = P @ M @ P # does the projection but doesnt reduce the structure dimension
		# 	P2 = P[~np.all(P == 0, axis=1)] # removes all fully zero rows

		# 	return P2 @ Mprime @ P2.transpose() # reduces the structure dimension
		return None

	# generates the embedding results for various time values
	def genEmbedOverTime(self, g:int, times:list[float]) -> tuple[list[float], tuple[np.ndarray, np.ndarray]]:
		H = self.H_embedding(g).full() # convert to numpy array
		results_embedded = [scipy.linalg.expm(-1j * t * H) for t in times]
		results_embedded_projected = [self.restrictToS(result) for result in results_embedded]

		results_original = [scipy.linalg.expm(-1j * t * self.A) for t in times]

		return (times, list(zip(results_embedded_projected, results_original)))

	def genEmbedOverG(self, gs:list[float], time:float) -> tuple[list[float], tuple[np.ndarray], np.ndarray]:
		results_embedded = [scipy.linalg.expm(-1j * time * self.H_embedding(g).full()) for g in gs]
		results_embedded_projected = [self.restrictToS(result) for result in results_embedded]

		results_original = [scipy.linalg.expm(-1j * time * self.A) for _ in gs]

		return (gs, list(zip(results_embedded_projected, results_original)))

	def genFrobAlone(self, results):
		return (results[0], [frobeniusNorm(emb) for (emb, _) in results[1]], [frobeniusNorm(orig) for (_, orig) in results[1]])

	def genSpectralAlone(self, results):
		return (results[0], [spectralNorm(emb) for (emb, _) in results[1]], [spectralNorm(orig) for (_, orig) in results[1]])

	# they said to use fidelity, which this norm encapsulates
	def genFrobBetween(self, results:tuple[list[float], tuple[np.ndarray, np.ndarray]]):
		return (results[0], [frobeniusNorm(emb - orig) for (emb, orig) in results[1]])

	# the comparison to use by default
	# since the Hamiltonian simualtion problem is defined using the spectral norm
	def genSpectralBetween(self, results:tuple[list[float], tuple[np.ndarray, np.ndarray]] = None):
		return (results[0], [spectralNorm(emb - orig) for (emb, orig) in results[1]])
		
	# will generate a matplotlib png or svg (TBD) of the fidelity results
	# can set a save path, if left blank, it will display the image instead
	# currently the image save argument does nothing
	def plotNormBetween(self, results:tuple[list[int], list[float]], xAxis:str, ham_type:str, pen_coef:int, normType:str, axLimits:tuple[float, float, float, float]=None, saveTopDirName:str=None):
		fig, ax = plt.subplots()
		ax.plot(results[0], results[1])
		normType = "Norm" if normType == None else (f"{normType} Norm")

		ax.set_xlabel(xAxis)
		ax.set_ylabel(normType)
		ax.set_title(f"{ham_type} Embedding with Penalty Coef. = {pen_coef}")

		if saveTopDirName != None:
			plt.savefig(f"{saveTopDirName}/{ham_type}-{pen_coef}-{normType}-{xAxis}-{axLimits}.png")

		plt.show()
		return ax

	# pass results as a (times/gs, embNorm, origNorm)
	def plotNormSeperate(self, results: tuple[list[float], list[float], list[float]], xAxis:str, ham_type:str, pen_coef:int, normType:str, axLimits:tuple[float, float, float, float]=None, saveTopDirName:str = None):
		fix, ax = plt.subplots()
		ax.plot(results[0], results[1])
		ax.plot(results[0], results[2])

		normType = "Norm" if normType == None else (f"{normType} Norm")

		ax.set_xlabel(xAxis)
		ax.set_ylabel(normType)
		ax.set_title(f"{ham_type} Embedding with Penalty Coef. = {pen_coef}")
	
		if axLimits != None:
			ax=plt.gca()
			ax.set_xlim([axLimits[0], axLimits[1]])
			ax.set_ylim([axLimits[2], axLimits[3]])

		if saveTopDirName != None:
			plt.savefig(f"{saveTopDirName}/{ham_type}-{pen_coef}-{normType}-{xAxis}-{axLimits}.png")

		plt.show()
		return ax

# Unary embedding
# this is an n-1 qubit operator
class UnaryEmbedding(HamiltonianEmbedding):

	# end at n-1 since python sums are not inclusive
	# while mathematical summations are
	# Pre: A is a square n x n array
	# Post: HPenUnary corresponding to A
	def HPen(self) -> np.ndarray:
		# note that this is dim-1 not dim-2 because python loops are
		# exclusive, while mathematical summations are inclusive
		n = self.dim
		H = -1 * sum(sub_i(Z, j+1, n-1) * sub_i(Z, j, n-1) + sub_i(Z, 1, n-1) - sub_i(Z, n-1, n-1)
		   for j in range(1, n- 1))
		
		return H

	# Pre: A is a square n x n array
	# Post: Q_A corresponding to A
	def Q_a(self) -> np.ndarray:
		# the usage of (act on qubit 1) is arbitraty and irrelevant
		# again note that the upper bounds of the summations 
		# are one higher than that of the paper
		# all index access into A is one lower than that of the paper because of computer science counting
		n = self.dim

		Q1 = (self.A[0, 0] * sub_i(I, 1, n-1) + sum((self.A[j-1][j-1] - self.A[j-2][j-2]) * sub_i(n_hat, j-1, n-1)
										 for j in range(2, n+1))) 
		Q2 = sum(sum(reduce(mat_mult, ([sub_i(X, l, n-1) for l in reversed(range(j+1, k))] + # j+1 to k-1, but +1 bc python
								 												   			 # also decrementing loop as in paper
								 [self.A[j-1][k-1].real * sub_i(X, j, n-1) - self.A[j-1][k-1].imag * sub_i(Y, j, n-1)]))	
			for j in range(1, k)) # since k is exclusive in paper, leaves it as is
			for k in range(2, n+1)) # +1 since python for loops are exclusive
		
		return (Q1 + Q2)

	def restrictToS(self, M:np.ndarray) -> np.ndarray:
		assert isinstance(M, np.ndarray)
		n = self.dim
		
		# getting the bitstrings that form the basis vectors for S
		# the reversal makes the bitstrings align with table 8 and the
		# little endian defintion given in Def. 3 under B.1.2 on pg. 39
		bitstrings = get_bitstrings_1d(n, "unary")
		# bitstrings = ["".join(reversed(bitstring)) for bitstring in bitstrings] ## removing this line, as per email

		uks = [qutip.tensor(qutip.basis(2, 0) if s == "0" else qutip.basis(2, 1) for s in bitstring) for bitstring in bitstrings]

		sBasisMatrix = np.column_stack([u.full() for u in uks])
		return (sBasisMatrix.transpose().conj() @ M @ sBasisMatrix)

		## deprecated by me
		# def ProjM(self) -> np.ndarray:
		# 	n = self.dim
		# 	bitstrings = get_bitstrings_1d(n, "unary")
		# 	print(bitstrings)
		# 	uks = [qutip.tensor(qutip.basis(2, 0) if s == "0" else qutip.basis(2, 1) for s in bitstring) for bitstring in bitstrings]
		# 	return sum([(u @ u.dag()) for u in uks])


# this is again an n-1 qubit operator
class AntiferromagneticEmbedding(HamiltonianEmbedding):

	# Eq. (B.13) from paper
	def HPen(self) -> np.ndarray:
		n = self.dim
		H = sum((sub_i(Z, j+1, n-1) @ sub_i(Z, j, n-1)) + sub_i(Z, 1, n-1) + (((-1)**(n-1)) * sub_i(Z, n-1, n-1))
		  for j in range(1, n-1)) # sigma j = 1, n-2, but +1 because of python range

		return H

	# Eq. (B.18) from the paper	
	def Q_a(self) -> np.ndarray:
		n = self.dim

		gamma = sum( ((-1)**(j+1)) * self.A[j-1][j-1] # alpha_j but cs index (i - 1)
			  for j in range(1, n+1)) # sigma j = 1 to n, but +1 bc python

		Q1 = (gamma * sub_i(I, 1, n-1)) + sum((((-1)**j) * (self.A[j-1][j-1] - self.A[j-2][j-2]) * sub_i(n_hat, j-1, n-1)) 
										for j in range(2, n+1)) # sigma j=2 to n, but +1 bc python
																# and alpha_j but -1 bc cs index

		Q2 = sum(sum(reduce(mat_mult, ([sub_i(X, l, n-1) for l in reversed(range(j+1, k))] + # j+1 to k-1, but +1 bc python
								 															 # reversed operator order as in paper
								[self.A[j-1][k-1].real * sub_i(X, j, n-1) - self.A[j-1][k-1].imag * sub_i(Y, j, n-1)]))
			   for j in range(1, k))
		   for k in range(2, n+1)) # sum 1 <= j < k <= n, but +1 bc python

		return (Q1 + Q2)


	def restrictToS(self, M:np.ndarray) -> np.ndarray:
		assert isinstance(M, np.ndarray)
		n = self.dim
		
		# getting the bitstrings that form the basis vectors for S
		# the reversal makes the bitstrings align with table 8 and the
		# little endian defintion given in Def. 3 under B.1.2 on pg. 39
		bitstrings = get_bitstrings_1d(n, "antiferromagnetic")
		bitstrings = ["".join(reversed(bitstring)) for bitstring in bitstrings]

		aks = [qutip.tensor(qutip.basis(2, 0) if s == "0" else qutip.basis(2, 1) for s in bitstring) for bitstring in bitstrings]

		sBasisMatrix = np.column_stack([a.full() for a in aks])
		return (sBasisMatrix.transpose().conj() @ M @ sBasisMatrix)
		

		## the restrictToS function should be used instead
		# page 40, under Eq. (B.25) of paper
		# returns the projection matrix for the antiferromagnetic embedding
		# def ProjM(self) -> np.ndarray: 
		# 	n = self.dim
		# 	bitstrings = (get_bitstrings_1d(n, "antiferromagnetic"))
		# 	print(f"bitstrings {bitstrings}")
		# 	aks = [qutip.tensor(qutip.basis(2, 0) if s == "0" else qutip.basis(2, 1) for s in bitstring) for bitstring in bitstrings]

		# 	return sum([(a @ a.dag()) for a in aks]) 



# nxn A -> n-qubit operator	
class OneHotEmbedding(HamiltonianEmbedding):

	# as in Eq. (B.43) of paper
	def HPen(self):
		n = self.dim
		rootH = sum((sub_i(n_hat, j, n) - 1)
			  for j in range(1, n+1)) # sum j=1 to n, but +1 bc python

		return rootH @ rootH # sqrt(H) @ sqrt(H) = H

	# as in Eq. (B.52) from the paper
	def Q_a(self):
		n = self.dim

		Q1 = sum(self.A[j-1][j-1] * sub_i(n_hat, j, n) # A_jj but -1 bc cs index
		   for j in range(1, n+1)) # sum j=1 to n, but +1 bc python

		Q2 = sum(sum((self.A[j-1][k-1].real * sub_i(X, k, n) @ sub_i(X, j, n)) + 
			   (self.A[j-1][k-1].imag * sub_i(X, k, n) @ sub_i(Y, j, n))
			   for j in range(1, k))
		   for k in range(2, n+1)) # sum 1 <= j < k <= n, but python range
		
		return (Q1 + Q2)

	def restrictToS(self, M:np.ndarray) -> np.ndarray:
		assert isinstance(M, np.ndarray)
		n = self.dim
		
		# getting the bitstrings that form the basis vectors for S
		# the reversal makes the bitstrings align with table 8 and the
		# little endian defintion given in Def. 3 under B.1.2 on pg. 39
		bitstrings = get_bitstrings_1d(n, "one-hot")
		bitstrings = ["".join(reversed(bitstring)) for bitstring in bitstrings]

		uks = [qutip.tensor(qutip.basis(2, 0) if s == "0" else qutip.basis(2, 1) for s in bitstring) for bitstring in bitstrings]

		sBasisMatrix = np.column_stack([u.full() for u in uks])
		return (sBasisMatrix.transpose().conj() @ M @ sBasisMatrix)

		## deprecated by me
		# guess based upon the other encodings
		# should be related to Def. 6
		# I think this is correct, because it preserved the 
		# P_SQ_AP_S = A identity
		# def ProjM(self):
		# 	n = self.dim
		# 	bitstrings = get_bitstrings_1d(n, "one-hot")
		# 	hks = [qutip.tensor(qutip.basis(2, 0) if s == "0" else qutip.basis(2, 1) for s in bitstring) for bitstring in bitstrings]
		# 	return sum([(h @ h.dag()) for h in hks])



# this has Hembed = HPen = Q_A
# so will override the HamiltonianEmbedding H_embedding() function.
# nxn A -> n-qubit operator
# note that there is no g (equiv to g = 1)
# so when calling H_embedding(g=), the value for g will be ignored
# this will make compatability of code easy (I hope)
class OneHotPenaltyFreeEmbedding(HamiltonianEmbedding):

	# there is no Hpen for the penalty free embeddding (hence the name)
	def HPen(self):
		return 0

	# Eq. (B.65) from paper
	def Q_a(self):
		n = self.dim

		H1 = sum(self.A[j-1][j-1] * sub_i(n_hat, j, n) # -1, -1 on index bc cs index
		   for j in range(1, n+1)) # sum j=1 to n, but +1 bc python
		
		H2 = sum(sum(self.A[j-1][k-1].real * ((sub_i(X, k, n) @ sub_i(X, j, n)) + (sub_i(Y, k, n) @ sub_i(Y, j, n)))
			   + self.A[j-1][k-1].imag * ((sub_i(X, k, n) @ sub_i(Y, j, n)) - (sub_i(Y, k, n) @ sub_i(X, j, n)))

			   for j in range(1, k))
		   for k in range(2, n+1)) # sum 1 <= j < k <= n, but +0, +1 python loops

		return (H1 + 0.5 * H2)

	def restrictToS(self, M:np.ndarray) -> np.ndarray:
		assert isinstance(M, np.ndarray)
		n = self.dim
		
		# getting the bitstrings that form the basis vectors for S
		# the reversal makes the bitstrings align with table 8 and the
		# little endian defintion given in Def. 3 under B.1.2 on pg. 39
		bitstrings = get_bitstrings_1d(n, "one-hot")
		bitstrings = ["".join(reversed(bitstring)) for bitstring in bitstrings]

		uks = [qutip.tensor(qutip.basis(2, 0) if s == "0" else qutip.basis(2, 1) for s in bitstring) for bitstring in bitstrings]

		sBasisMatrix = np.column_stack([u.full() for u in uks])
		return (sBasisMatrix.transpose().conj() @ M @ sBasisMatrix)

		## deprecated by me
		# # same ProjM as the OneHotEmbedding	class
		# def ProjM(self):
		# 	n = self.dim
		# 	bitstrings = get_bitstrings_1d(n, "one-hot")
		# 	hks = [qutip.tensor(qutip.basis(2, 0) if s == "0" else qutip.basis(2, 1) for s in bitstring) for bitstring in bitstrings]
		# 	return sum([(h @ h.dag()) for h in hks])



