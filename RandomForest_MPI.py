#IMPORTS:

import numpy as np
from sklearn.base import BaseEstimator
from mpi4py import MPI
import pickle
import argparse as ag

#IMPURITIES:

def gini(labels):
    p = np.bincount(labels)/len(labels)
    return np.dot(p, 1-p)

def entropy(labels):
    p = np.bincount(labels)/len(labels)
    return -np.dot(p, np.log2(p))

def misclass_err(labels):
    p = np.bincount(labels)/len(labels)
    try:
        return 1 - max(p)
    except:
        return 0

class Node:
    
    def __init__(self, indices, max_depth, parent=None):
        
        self.max_depth = max_depth
        if parent == None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1
        
        if self.depth >= self.max_depth:
            self.is_leaf = True
        else:
            self.is_leaf = False
        
        self.indices = indices  
            
    def extend(self, stack, impurity, data, labels, feature_set):
        
        labs = labels[self.indices]
        
        if self.is_leaf:
            self.size = len(labs)
            try:
                self.prob = (np.bincount(labs)/len(labs))[1]
            except:
                self.prob = 0
            return 
        
        data_set = data[self.indices, :]
        
        max_gain = np.NINF
        best_j = None
        best_t = None
        best_lind = None
        best_rind = None
        
        for j in feature_set:
                
            vals = np.unique(data_set[:, j])
            ths = np.pad((vals[:-1] + vals[1:])/2, (1, 1), mode='constant', 
                         constant_values=(vals[0], vals[-1]+0.0001))
            for t in ths:
                node_imp = (data_set.shape[0]/data.shape[0])*impurity(labs)
                rind = np.logical_and(self.indices, data[:, j] >= t)
                right_imp = (rind.sum()/data.shape[0])*impurity(labels[rind])
                lind = np.logical_and(self.indices, data[:, j] < t)
                left_imp = (lind.sum()/data.shape[0])*impurity(labels[lind])
                gain = node_imp - (right_imp + left_imp)
                assert(np.logical_xor(rind, lind).sum() == data_set.shape[0])
                if gain > max_gain:
                    max_gain = gain
                    best_j = j
                    best_t = t
                    best_lind = lind
                    best_rind = rind
                    
        if max_gain > 0:
            self.feature = best_j
            self.threshold = best_t
            self.left_child = Node(best_lind, self.max_depth, self)
            self.right_child = Node(best_rind, self.max_depth, self) 
            stack.append(self.left_child)
            stack.append(self.right_child)
        else:
            self.is_leaf = True
            self.size = len(labs)
            try:
                self.prob = (np.bincount(labs)/len(labs))[1]
            except:
                self.prob = 0
    
    def decide(self, example):
        if self.is_leaf:
            return self.prob
        else:
            if example[self.feature] >= self.threshold:
                return self.right_child
            else:
                return self.left_child
            
    def prunify(self):
        
        if not self.is_leaf:
            left_p, left_s = self.left_child.prunify()
            right_p, right_s = self.right_child.prunify()
            self.size = left_s + right_s
            self.prob = (left_p*left_s+right_p*right_s)/self.size          
            
        return self.prob, self.size
            
    '''        
    def BFS_prob(self):
        probs = []
        queue = [self]
        while queue:
            curr = queue.pop(0)
            if curr.is_leaf:
                prob.append(self.prob)
            else:
                pass
        return np.mean(probs)
    '''
    
    def decide_prunned(self, example, new_depth):
        if (self.depth > new_depth):
            raise ValueError('Depth exceeded')
            
        if (self.is_leaf) or (self.depth == new_depth):
            return self.prob
        else:
            if example[self.feature] >= self.threshold:
                return self.right_child
            else:
                return self.left_child

class Decision_Tree(BaseEstimator):
    
    def __init__(self, impurity, max_depth=5):
        self.impurity = impurity
        self.max_depth = max_depth
    
    def fit(self, data_set, labels, feature_set):
        self.root = Node(np.full(data_set.shape[0], True), max_depth=self.max_depth)
        self.NodeStack = [self.root]
        while self.NodeStack:
            curr = self.NodeStack.pop()
            curr.extend(self.NodeStack, self.impurity, data_set, labels, feature_set)
            
    def predict_proba(self, data):
        probas = np.zeros([data.shape[0], 2])
        for i, x in enumerate(data):
            curr = self.root
            while not curr.is_leaf:
                curr = curr.decide(x)
            p = curr.decide(x)
            probas[i, :] = [1 - p, p]
        return probas
    
    def prunify(self):
        self.root.prunify()
        self.prunified = True
    
    def predict_proba_prunned(self, data, new_depth):
        
        if not self.prunified:
            raise ValueError('Tree was not prunified')
            
        probas = np.zeros((data.shape[0], 2))
        for i, x in enumerate(data):
            curr = self.root
            while (not curr.is_leaf) and (curr.depth != new_depth):
                curr = curr.decide_prunned(x, new_depth)
            p = curr.decide_prunned(x, new_depth)
            probas[i, :] = [1 - p, p]
        return probas
    
    def predict(self, data):
        probas = self.predict_proba(data)
        preds = probas[:, 1] > 0.5
        return preds
    
class Random_Forest:
    
    def __init__(self, n_trees, max_depth, feature_size, impurity=gini):
        
        self.n_trees = n_trees
        self.forest = None
        self.impurity = impurity
        self.max_depth = max_depth
        self.feature_size = feature_size

    def fit(self, data_set, labels):
        
        fsize = int(data_set.shape[1] * self.feature_size)
               
        core_trees = []
        
        i = comm.rank
        while i < self.n_trees:
            
            bootstrap = np.random.choice(data_set.shape[0], data_set.shape[0], replace=True)
            feature_set = np.random.choice(data_set.shape[1], fsize, replace=False)
            b_data = data_set[bootstrap]
            b_labels = labels[bootstrap]
            
            model = Decision_Tree(self.impurity, self.max_depth)
            model.fit(b_data, b_labels, feature_set)
            core_trees.append(model)
            
            i += comm.size
            
        comm.Barrier()
        
        self.forest = comm.gather(core_trees, root=0)
        if comm.rank == 0:
            self.forest = [tree for subforest in self.forest for tree in subforest]

    def predict_proba(self, data):
        probas = np.zeros((data.shape[0], self.n_trees))
        for i in range(self.n_trees):
            est = self.forest[i]
            p = est.predict_proba(data)
            probas[:, i] = p[:, 1]
        probas = probas.mean(axis=1)
        #return np.vstack([1 - probas, probas]).T #A problem may have arised here, be caustious
        return probas

    def predict(self, data):
        pass
        
    def prunify(self):
        for tree in self.forest:
            tree.prunify()
            
    def predit_proba_prunned(self, data, new_ntrees, new_depth):
        
        if (new_ntrees == self.n_trees) and (new_depth == self.max_depth):
            probas = self.predict_proba(data)
            return probas
        
        tree_sample = np.random.choice(self.n_trees, size=new_ntrees, replace=False)
        probas = np.zeros((data.shape[0], new_ntrees))
        for i, ti in enumerate(tree_sample):
            est = self.forest[ti]
            p = est.predict_proba_prunned(data, new_depth)
            probas[:, i] = p[:, 1]
        probas = probas.mean(axis=1)
        return probas

    def get_smaller_tree(self, new_ntrees):
        pass
    
def parse_args():
    parser = ag.ArgumentParser()
    parser.add_argument('-d', '-data', required=True, dest='data', type=str, help='Path to data')
    parser.add_argument('-l', '-labels', required=True, dest='labs', type=str, help='Path to labels')
    parser.add_argument('-nt', '-number-of-trees', required=True, dest='n_trees', 
                        type=int, help='Number of trees in the forest')
    parser.add_argument('-md', '-max-depth', required=True, dest='max_depth', 
                        type=int, help='Maximum depth of a single tree')
    #parser.add_argument('-ss', '-sample-size', required=True, dest='sample_size', 
    #                   type=float, help='Size of a random subsample (0, 1]')
    parser.add_argument('-fs', '-feature-size', required=True, dest='feature_size', 
                        type=float, help='Size of a random feature set')
    parser.add_argument('-o', '-out', required=True, dest='out',
                        type=str, help='Output file for a pickle object containing model')
    args = vars(parser.parse_args())
    return args
    
if __name__ == "__main__":
    args = parse_args()
    #MAYBE YOU SHOLD MAKE IT PARALLEL ALREADY THERE?
    comm = MPI.COMM_WORLD
    
    data = None
    labels = None
    
    if comm.rank == 0:
        data = np.load(args['data'])
        labels = np.load(args['labs'])
        
    data = comm.bcast(data)
    labels = comm.bcast(labels)
    #print(args['n_trees'], args['max_depth'], args['feature_size'])
    model = Random_Forest(args['n_trees'], args['max_depth'], args['feature_size'])
    model.fit(data, labels)
    comm.Barrier()
    
    if comm.rank == 0:
        with open(args['out'], 'wb') as fn:
            pickle.dump(model, fn)