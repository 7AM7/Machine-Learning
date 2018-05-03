import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
style.use('ggplot')

class Support_Vecotr_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.color = {1:'r', -1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

    #train
    def fit(self, data):
        self.data = data
        # { ||w||: [w,b] }
        opt_dict = {} ## ||w|| = sqrt(x**2 + y**)
        
        '''
        transforms = [[1,1], ## trying only 4 rotations of w (90 degrees)
                      [-1,1],
                      [-1,-1],
                      [1,-1]]
        '''
       ## Reference : https://scipython.com/book/chapter-6-numpy/examples/creating-a-rotation-matrix-in-numpy/
       ## R=(cosθ  -sinθ
       ##     sinθ   cosθ)
       ## trying infinte rotations 
        rotMatrix = lambda theta: np.array([[np.cos(theta), -np.sin(theta)], 
                         [np.sin(theta),  np.cos(theta)]])
        
        thetaStep = np.pi/20 ## get theta step
        transforms = [ (np.matrix(rotMatrix(theta)) * np.matrix([1,0]).T).T.tolist()[0]
                       for theta in np.arange(0,np.pi,thetaStep) ]
        #print(transforms)

        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        ## free list
        all_data = None

        # support vectors yi(xi.w+b) = 1
        ## move with big stpes if we find best point we will move with small
        ## then move with more small steps


        # SVM yi(xi.w+b) = 1 

        step_sizes = [self.max_feature_value * 0.1,## take a big step
                      self.max_feature_value * 0.01,## take small step
                      # point of expense:
                      self.max_feature_value * 0.001,## take more small step
                      ]

        
        
        # extremely expensive
        b_range_multiple = 2
        # we dont need to take as small of steps
        # with b as we do w
        b_multiple = 5
        latest_optimum = self.max_feature_value*10
        
        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum])
            
            ## we do this beacuse convex
            ## we don't want to move more steps if get best point
            ## but we can take a some more step for check too.
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),# from
                                   self.max_feature_value*b_range_multiple,# to
                                   step*b_multiple):# how much step we will move
                    for transformation in transforms:
                        w_t = w*transformation # W vector after transformation
                        found_option = True
                        # weakest link in the SVM fundamentally
                        # SMO attempts to fix this a bit
                        # yi(xi.w+b) >= 1
                        for i in self.data:# i as yi
                            for xi in self.data[i]:## xi as a features
                                yi=i
                                if not yi*(np.dot(w_t,xi)+b) >= 1: 
                                    found_option = False
                                    break
                            if not found_option:
                                break
                                    
                        if found_option:
                            ##np.linalg.norm(w_t) = np.sqrt(np.sum((np.array(w_t[0]) + np.array(w_t[1]))**2))
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b] ## ||W||

                if w[0] < 0: ## if x in vector w less then 0 we get the good point
                    optimized = True
                    print('Optimized a step.')
                else:
                    ## w= [5,5]
                    # step = 1
                    # w - step = [4,4]
                    w = w - step
                    
            
            norms = sorted([n for n in opt_dict]) 
            #print(norms)
            #print(opt_dict)
            #||w|| : [w,b]
            ## remmeber the opt_dict = ||W||:[w,b]
            opt_choice = opt_dict[norms[0]]# get the value [w,b]
            self.w = opt_choice[0] # get the value of W(x,y) as W is vector
            self.b = opt_choice[1]  # get the value of b
            latest_optimum = opt_choice[0][0]+step*2
               
    def predict(self, features):
        #sign( x.w+b )
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.color[classification])
            
        return classification

    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=100, color=self.color[i]) for x in data_dict[i]] for i in data_dict]
        # hyperplane = x.w+b
        # v = x.w +b
        # postive support vector = 1
        # negative support vector = -1
        # decislon boundary support vector = 0 
        def hyperplane(x, w, b, v):
            return (-w[0]*x-b+v) / w[1]

        data_range = (self.min_feature_value*0.9, self.max_feature_value*1.1)
        hyp_x_min = data_range[0]
        hyp_x_max = data_range[1]

        # (w.x+b) = 1
        # positve suppoort vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1) ## get y 
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2])

        # (w.x+b) = -1
        # negative suppoort vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1) ## get y 
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2])
        
        # (w.x+b) = 0
        # decislon boundary suppoort vector hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0) ## get y 
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2],'y--')

        plt.show()


data_dict = {-1:np.array([[1,7],
                         [2,8],
                         [3,8],]),

             1:np.array([[5,-1],
                         [6,-1],
                         [7,3],])}

svm = Support_Vecotr_Machine()
svm.fit(data=data_dict)

predict_us = [[0,10],
              [1,3],
              [3,4],
              [3,5],
              [5,5],
              [5,6],
              [6,-5],
              [5,8],]

for p in predict_us:
    svm.predict(p)
svm.visualize()
