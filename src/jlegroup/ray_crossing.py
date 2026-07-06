# -*- coding: utf-8 -*-
"""
Created on Sun Sep 9 2021

@author: WataThep
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate

class RayCrossing:
    
    '''Map position in observer's plane to planetary plane for each point in the "position" array
    r_array (array of float): position in planetary plane
    bendingAngle (array of float): bending angle corresponding to each position in r_array
    distance(float): distance from the planet to the observer
    position (array of float): position in observer's plane at which the data are taken'''
    
    def __init__(self, r_array, bendingAngle, distance, position):
        
        self.x = r_array
        self.theta = bendingAngle
        self.d = distance
        self.position = position
        
        self.y = None
        self.range_list = None
        self.func_list = None
        self.r = None
        
    def getConversionFunction(self):
        
        #convert r to y
        
        y = self.x+self.d*self.theta
        self.y = y
        
        #check for the position where y change from increasing to dcreasing (or vice versa)
        #this will be the boundary of each segment
        
        index = [0]
        for i in range (1,len(self.x)-1):
            if self.y[i-1] > self.y[i] and self.y[i] < self.y[i+1]:
                index.append(i)
            elif self.y[i-1] < self.y[i] and self.y[i] > self.y[i+1]:
                index.append(i)
        
        index.append(len(self.x)-1)
        
        if len(index)==2:
            
            #No ray-crosing
            
            self.range_list = [np.array([-np.inf, np.inf])]
            self.func_list = [interpolate.CubicSpline(self.y,self.x)]
            
        else:
            
            #range of y for each segment
            range_list = [np.sort([self.y[index[i]], self.y[index[i+1]]]) for i in range (1,len(index)-2)]
            
            #now we can add the first segment
            if self.y[0]<self.y[1]:
                #y-values in the first segment is increasing
                range_list = [np.sort([-np.inf,self.y[index[1]]])]+range_list
            else:
                #this should not happen, but just for numerical stuff
                range_list = [np.sort([np.inf,self.y[index[1]]])]+range_list
            
            #add the last segment
            if y[-1]<y[-2]:
                #y-value in the last segment is decreasing, which should not happen
                range_list = range_list+[np.sort([self.y[index[-2]], -np.inf])]
            else:
                range_list = range_list+[np.sort([self.y[index[-2]], np.inf])]
            
            self.range_list = range_list
    
            #Build a spline for each segment
            func_list = []
            
            for i in range (0,len(index)-1):
                df = pd.DataFrame({'y':self.y[index[i]:index[i+1]],
                                   'x':self.x[index[i]:index[i+1]]})
                df=df.sort_values('y')
                func_list.append(interpolate.CubicSpline(df['y'],df['x']))
            
            self.func_list = func_list
    
    def main(self):
        
        self.getConversionFunction()
        
        r = []
        
        #loop through each point in position array and find out where in the planetary plane the light rays come from
        for e in self.position:
            
            temp = []
            
            for i in range(0,len(self.range_list)):
                #check whether this point is in each segment
                if self.range_list[i][0]<self.range_list[i][1] and e>self.range_list[i][0] and e<self.range_list[i][1]:
                    temp.append(self.func_list[i](e))
                elif self.range_list[i][0]>self.range_list[i][1] and e<self.range_list[i][0] and e>self.range_list[i][1]:
                    temp.append(self.func_list[i](e))
            r.append(temp)
        self.r = r