#!/usr/bin/env python

from __future__ import print_function

import argparse
import logging

import rosbag_pandas
import numpy as np
import random

import pykst as kst
from scipy.signal import butter, lfilter, freqz

def build_parser():
    """
    Builds the parser for reading the command line arguments
    :return: Argument parser
    """
    parser = argparse.ArgumentParser(description='Print one or multiple bag keys')
    parser.add_argument('-b', '--bag', help='Bag file to read',
                        required=True, nargs='*')
    parser.add_argument('-k', '--key',
                        help='Key you would like to print',
                        required=False, nargs='*')
    parser.add_argument('-v', '--verbose',
                        help="Log verbose",
                        default=False, action="store_true")
    parser.add_argument('-t', '--type',
                        help="Type of graph: e: error, r: rpm, d: drag force, j: jerk",
                        default='e', type=str)   
    parser.add_argument('-r', '--random',
                        default=1.0,type=float)

    return parser

def reindex_up(a,b):
    #print("a",len(a.index),"b",len(b.index))
    if (len(a.index) > len(b.index)):
        b = b.reindex(index = a.index, method = 'bfill')
        a = a.fillna(method="bfill")
    else:
        a = a.reindex(index = b.index, method = 'bfill')
        b = b.fillna(method="bfill")
    return [a,b]

def reindex_down(a,b):
    #print("a",len(a.index),"b",len(b.index))
    if (len(a.index) < len(b.index)):
        b = b.reindex(index = a.index, method = 'bfill')
        a = a.fillna(method="bfill")
    else:
        a = a.reindex(index = b.index, method = 'bfill')
        b = b.fillna(method="bfill")
    return [a,b]

def graph_accelerations(df1, channels):

    #df1['jx_avg'] = (df1[channels[0]].dropna()+df1[channels[0]].dropna().shift(1).fillna(0))*0.01/2
    #df1['jy_avg'] = (df1[channels[1]].dropna()+df1[channels[1]].dropna().shift(1).fillna(0))*0.01/2
    #df1['jz_avg'] = (df1[channels[2]].dropna()+df1[channels[2]].dropna().shift(1).fillna(0))*0.01/2
    df1['jx_avg'] = (df1[channels[0]])*0.01
    df1['jy_avg'] = (df1[channels[1]])*0.01
    df1['jz_avg'] = (df1[channels[2]])*0.01
    df1['jx_int'] =  df1['jx_avg']#.cumsum()
    df1['jy_int'] =  df1['jy_avg']#.cumsum()
    df1['jz_int'] =  df1['jz_avg']#.cumsum()

    j_int_x = np.array(df1[channels[0]].values)
    j_int_x_time = np.array(df1[channels[0]].index)
    j_int_y = np.array(df1[channels[1]].values)
    j_int_y_time = np.array(df1[channels[1]].index)
    j_int_z = np.array(df1[channels[2]].values)
    j_int_z_time = np.array(df1[channels[2]].index)

    """
    j_int_x = np.array(df1['jx_int'].values)
    j_int_x_time = np.array(df1['jx_int'].index)
    j_int_y = np.array(df1['jy_int'].values)
    j_int_y_time = np.array(df1['jy_int'].index)
    j_int_z = np.array(df1['jz_int'].values)
    j_int_z_time = np.array(df1['jz_int'].index)
    """

    a_x = np.array(df1[channels[3]].dropna().values)
    a_x_time = np.array(df1[channels[3]].dropna().index)
    a_y = np.array(df1[channels[4]].dropna().values)
    a_y_time = np.array(df1[channels[4]].dropna().index)
    a_z = np.array(df1[channels[5]].dropna().values)
    a_z_time = np.array(df1[channels[5]].dropna().index)

    p1 = client.new_plot()
    add_curves_to_kst([j_int_x_time,j_int_x,a_x_time,a_x],["time","j_int_x","time","a_x"],p1)
    p2 = client.new_plot()
    add_curves_to_kst([j_int_y_time,j_int_y,a_y_time,a_y],["time","j_int_y","time","a_y"],p2)
    p3 = client.new_plot()
    add_curves_to_kst([j_int_z_time,j_int_z,a_z_time,a_z],["time","j_int_z","time","a_z"],p3)

def shift5(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def graph_accelerations2(df1, channels):

    j_x = df1[channels[0]].dropna()
    j_y = df1[channels[1]].dropna()
    j_z = df1[channels[2]].dropna()

    a_x = df1[channels[3]].dropna()
    a_y = df1[channels[4]].dropna()
    a_z = df1[channels[5]].dropna()

    

    
    #j_x_int = 0.01*(j_x + j_x.shift(-1))/2
    #j_y_int = 0.01*(j_y + j_y.shift(-1))/2
    #j_z_int = 0.01*(j_z + j_z.shift(-1))/2

    j_x_int = (j_x*0.01).cumsum()
    j_y_int = (j_y*0.01).cumsum()
    j_z_int = (j_z*0.01).cumsum()

    """
    v_x = df1[channels[6]].dropna()
    v_y = df1[channels[7]].dropna()
    v_z = df1[channels[8]].dropna()

    w_x = df1[channels[9]].dropna()
    w_y = df1[channels[10]].dropna()
    w_z = df1[channels[11]].dropna()

    v_mag = euclidean_magnitude(v_x, v_y, v_z)
    w_mag = euclidean_magnitude(w_x, w_y, w_z)

    v_vals = np.array(v_mag.values)
    v_time = np.array(v_mag.index - v_mag.index[0])
    w_vals = np.array(w_mag.values)
    w_time = np.array(w_mag.index - w_mag.index[0])
    """
    
    j_x_vals = np.array(j_x_int.values)
    j_x_time = np.array(j_x_int.index - j_x_int.index[0])
    j_y_vals = np.array(j_y_int.values)
    j_y_time = np.array(j_y_int.index - j_y_int.index[0])
    j_z_vals = np.array(j_z_int.values)
    j_z_time = np.array(j_z_int.index - j_z_int.index[0])

    a_x_vals = np.array(a_x.values)
    a_x_time = np.array(a_x.index - a_x.index[0])
    a_y_vals = np.array(a_y.values)
    a_y_time = np.array(a_y.index - a_y.index[0])
    a_z_vals = np.array(a_z.values)
    a_z_time = np.array(a_z.index - a_z.index[0])

    #a1, b1 = reindex_down(j_x_int, a_x)
    #a2, b2 = reindex_down(j_y_int, a_y)
    #a3, b3 = reindex_down(j_z_int, a_z)
    
    # metric: C = a - integral(j)
    last = min(len(a_x_vals),len(j_x_vals)) - 1 
    c_x = a_x_vals[:last] - j_x_vals[:last]
    c_y = a_y_vals[:last] - j_y_vals[:last]
    c_z = a_z_vals[:last] - j_z_vals[:last]
    


    a_mag = euclidean_magnitude(a_x_vals[:last],a_y_vals[:last],a_z_vals[:last])
    j_mag = euclidean_magnitude(j_x_vals[:last],j_y_vals[:last],j_z_vals[:last])
    c_mag = euclidean_magnitude(c_x, c_y, c_z)
    print("AJE: {}".format(jae(c_x, c_y, c_z)))


    """
    # metric: dC/dt = v - j
    j_x2, v_x2 = reindex_down(j_x, v_x)
    j_y2, v_y2 = reindex_down(j_y, v_y)
    j_z2, v_z2 = reindex_down(j_z, v_z)  

    j_x2_vals = np.array(j_x2.values)
    j_x2_time = np.array(j_x2.index - j_x2.index[0])
    j_y2_vals = np.array(j_y2.values)
    j_y2_time = np.array(j_y2.index - j_y2.index[0])
    j_z2_vals = np.array(j_z2.values)
    j_z2_time = np.array(j_z2.index - j_z2.index[0])

    v_x2_vals = np.array(v_x2.values)
    v_x2_time = np.array(v_x2.index - v_x2.index[0])
    v_y2_vals = np.array(v_y2.values)
    v_y2_time = np.array(v_y2.index - v_y2.index[0])
    v_z2_vals = np.array(v_z2.values)
    v_z2_time = np.array(v_z2.index - v_z2.index[0])

    
    dc_x2 = np.array(map(lambda y2,y1,t2,t1: (y2-y1)/(t2-t1) if (t2-t1)>0.00001 else 0., j_x2_vals, shift5(j_x2_vals,1,0), j_x2_time, shift5(j_x2_time,1,0) ))
    dc_y2 = np.array(map(lambda y2,y1,t2,t1: (y2-y1)/(t2-t1) if (t2-t1)>0.00001 else 0., j_y2_vals, shift5(j_y2_vals,1,0), j_y2_time, shift5(j_y2_time,1,0) ))
    dc_z2 = np.array(map(lambda y2,y1,t2,t1: (y2-y1)/(t2-t1) if (t2-t1)>0.00001 else 0., j_z2_vals, shift5(j_z2_vals,1,0), j_z2_time, shift5(j_z2_time,1,0) ))

    # Filter requirements.
    order = 6
    fs = 100.0       # sample rate, Hz
    cutoff = 2    # desired cutoff frequency of the filter, Hz
    # Filter the data, and plot both the original and filtered signals.
    dc_x2 = butter_lowpass_filter(dc_x2, cutoff, fs, order)
    dc_y2 = butter_lowpass_filter(dc_y2, cutoff, fs, order)
    dc_z2 = butter_lowpass_filter(dc_z2, cutoff, fs, order)

    dc_x = v_x2_vals - j_x2_vals
    dc_y = v_y2_vals - j_y2_vals
    dc_z = v_z2_vals - j_z2_vals
    dc_x_mean = np.sum(dc_x)/len(dc_x)
    dc_y_mean = np.sum(dc_y)/len(dc_y)  
    dc_z_mean = np.sum(dc_z)/len(dc_z)
    print(dc_x_mean, dc_y_mean, dc_z_mean)  
    """

    p1 = client.new_plot(pos=(0.0,0.0))
    add_curves_to_kst([j_x_time,-j_x_vals,a_x_time,a_x_vals, j_x_time[:last], c_x],["time","\int j_x","time","a_x", "time","C_x"],p1)    
    p2 = client.new_plot(pos=(0.0,0.3))
    add_curves_to_kst([j_y_time,-j_y_vals,a_y_time,a_y_vals, j_y_time[:last], c_y],["time","\int j_y","time","a_y", "time","C_y"],p2)
    p3 = client.new_plot(pos=(0.0,0.6))
    add_curves_to_kst([j_z_time,j_z_vals,a_z_time,a_z_vals, j_z_time[:last], c_z],["time","\int j_z","time","a_z", "time","C_z"],p3)
    #p4 = client.new_plot(pos=(0.5,0.3))
    #add_curves_to_kst([v_time,v_vals,w_time,w_vals],["time","|v|","time","|\omega|"],p4)
    #print("v avg: {} m/s".format(np.mean(v_vals)))
    p5 = client.new_plot(pos=(0.5,0.0))
    add_curves_to_kst([j_x_time[:last],j_mag,j_x_time[:last],a_mag,j_x_time[:last],c_mag ],["time","|j|","time","|a|","time","|C|"],p5)

def add_curves_to_kst(pairs, name_pairs, plot):

    index = np.arange(0,len(pairs),2)

    for i in index:
        x1 = np.array(pairs[i])
        y1 = np.array(pairs[i+1])

        V1 = client.new_editable_vector(x1, name=name_pairs[i]) # the name is for the label
        V2 = client.new_editable_vector(y1, name=name_pairs[i+1]) # the name is for the label  

        c1 = client.new_curve(V1, V2)
        plot.add(c1)

def euclidean_magnitude(a,b,c):
    return np.sqrt(a**2 + b**2 + c**2)

def ate(x,y,z):
    e = x**2 + y**2 + z**2
    e = e.sum()
    e = e/len(x)
    e = np.sqrt(e)
    return e

def jae(x,y,z):
    e = (x - x.mean())**2 + (y - y.mean())**2 + (z - z.mean())**2
    e = e.sum()
    e = e/len(x)
    e = np.sqrt(e)
    return e

def rpm_ate(x,y,z,k): 
    e = x**2 + y**2 + z**2 + k**2
    e = e.sum()
    e = e/len(x)
    e = np.sqrt(e)
    return e

def graph_errors(df1, df2, channels, r):

    err_abs_x_no_jerk = df1[channels[0]].dropna()    
    err_abs_y_no_jerk = df1[channels[1]].dropna()
    err_abs_z_no_jerk = df1[channels[2]].dropna()

    err_abs_x_with_jerk = df2[channels[0]].dropna() * r   
    err_abs_y_with_jerk = df2[channels[1]].dropna() * r
    err_abs_z_with_jerk = df2[channels[2]].dropna() * r

    v_x1 = df1[channels[3]].dropna()
    v_y1 = df1[channels[4]].dropna()
    v_z1 = df1[channels[5]].dropna()

    w_x1 = df1[channels[6]].dropna()
    w_y1 = df1[channels[7]].dropna()
    w_z1 = df1[channels[8]].dropna()

    v_x2 = df2[channels[3]].dropna()
    v_y2 = df2[channels[4]].dropna()
    v_z2 = df2[channels[5]].dropna()

    w_x2 = df2[channels[6]].dropna()
    w_y2 = df2[channels[7]].dropna()
    w_z2 = df2[channels[8]].dropna()

    v_mag1 = euclidean_magnitude(v_x1, v_y1, v_z1)
    w_mag1 = euclidean_magnitude(w_x1, w_y1, w_z1)
    v_mag2 = euclidean_magnitude(v_x2, v_y2, v_z2)
    w_mag2 = euclidean_magnitude(w_x2, w_y2, w_z2)

    v_vals1 = np.array(v_mag1.values)
    v_time1 = np.array(v_mag1.index - v_mag1.index[0])
    w_vals1 = np.array(w_mag1.values)
    w_time1 = np.array(w_mag1.index - w_mag1.index[0])
    v_vals2 = np.array(v_mag2.values)
    v_time2 = np.array(v_mag2.index - v_mag2.index[0])
    w_vals2 = np.array(w_mag2.values)
    w_time2 = np.array(w_mag2.index - w_mag2.index[0])

    err_abs_mag_with_jerk = euclidean_magnitude(err_abs_x_with_jerk, err_abs_y_with_jerk, err_abs_z_with_jerk)
    err_abs_mag_no_jerk = euclidean_magnitude(err_abs_x_no_jerk, err_abs_y_no_jerk, err_abs_z_no_jerk)

    err_abs_x_no_jerk_vals = np.array(err_abs_x_no_jerk.values)
    err_abs_x_no_jerk_time = np.array(err_abs_x_no_jerk.index - err_abs_x_no_jerk.index[0])
    err_abs_y_no_jerk_vals = np.array(err_abs_y_no_jerk.values)
    err_abs_y_no_jerk_time = np.array(err_abs_y_no_jerk.index - err_abs_y_no_jerk.index[0])
    err_abs_z_no_jerk_vals = np.array(err_abs_z_no_jerk.values)
    err_abs_z_no_jerk_time = np.array(err_abs_z_no_jerk.index - err_abs_z_no_jerk.index[0])

    err_abs_x_with_jerk_vals = np.array(err_abs_x_with_jerk.values)
    err_abs_x_with_jerk_time = np.array(err_abs_x_with_jerk.index - err_abs_x_with_jerk.index[0])
    err_abs_y_with_jerk_vals = np.array(err_abs_y_with_jerk.values)
    err_abs_y_with_jerk_time = np.array(err_abs_y_with_jerk.index - err_abs_y_with_jerk.index[0])
    err_abs_z_with_jerk_vals = np.array(err_abs_z_with_jerk.values)
    err_abs_z_with_jerk_time = np.array(err_abs_z_with_jerk.index - err_abs_z_with_jerk.index[0])

    p1 = client.new_plot()
    add_curves_to_kst([err_abs_x_no_jerk_time,err_abs_x_no_jerk_vals,err_abs_x_with_jerk_time,err_abs_x_with_jerk_vals],["time","e_{x,flc}","time","e_{x,fc}"],p1)    
    p2 = client.new_plot()
    add_curves_to_kst([err_abs_y_no_jerk_time,err_abs_y_no_jerk_vals,err_abs_y_with_jerk_time,err_abs_y_with_jerk_vals],["time","e_{y,flc}","time","e_{y,fc}"],p2)
    p3 = client.new_plot()
    add_curves_to_kst([err_abs_z_no_jerk_time,err_abs_z_no_jerk_vals,err_abs_z_with_jerk_time,err_abs_z_with_jerk_vals],["time","e_{z,flc}","time","e_{z,fc}"],p3)
    p4 = client.new_plot()
    add_curves_to_kst([err_abs_z_no_jerk_time,err_abs_mag_no_jerk,err_abs_z_with_jerk_time,err_abs_mag_with_jerk],["time","||e_{flc}||","time","||e_{fc}||"],p4)

    print("ATE FLC: {:.5f} m".format(ate(err_abs_x_no_jerk, err_abs_y_no_jerk, err_abs_z_no_jerk)))
    print("FLC v avg: {} m/s".format(np.mean(v_vals1)))

    print("ATE FC: {:.5f} m".format(ate(err_abs_x_with_jerk, err_abs_y_with_jerk, err_abs_z_with_jerk)))
    print("FC v avg: {} m/s".format(np.mean(v_vals2)))


def graph_rpms(df1, df2, channels):

    err_abs_x_no_jerk = df1[channels[0]].dropna()    
    err_abs_y_no_jerk = df1[channels[1]].dropna()
    err_abs_z_no_jerk = df1[channels[2]].dropna()
    err_abs_k_no_jerk = df1[channels[3]].dropna()

    err_abs_x_with_jerk = df2[channels[0]].dropna()    
    err_abs_y_with_jerk = df2[channels[1]].dropna()
    err_abs_z_with_jerk = df2[channels[2]].dropna()
    err_abs_k_with_jerk = df2[channels[3]].dropna()

    err_abs_mag_with_jerk = euclidean_magnitude(err_abs_x_with_jerk, err_abs_y_with_jerk, err_abs_z_with_jerk)
    err_abs_mag_no_jerk = euclidean_magnitude(err_abs_x_no_jerk, err_abs_y_no_jerk, err_abs_z_no_jerk)

    err_abs_x_no_jerk_vals = np.array(err_abs_x_no_jerk.values)
    err_abs_x_no_jerk_time = np.array(err_abs_x_no_jerk.index - err_abs_x_no_jerk.index[0])
    err_abs_y_no_jerk_vals = np.array(err_abs_y_no_jerk.values)
    err_abs_y_no_jerk_time = np.array(err_abs_y_no_jerk.index - err_abs_y_no_jerk.index[0])
    err_abs_z_no_jerk_vals = np.array(err_abs_z_no_jerk.values)
    err_abs_z_no_jerk_time = np.array(err_abs_z_no_jerk.index - err_abs_z_no_jerk.index[0])
    err_abs_k_no_jerk_vals = np.array(err_abs_k_no_jerk.values)
    err_abs_k_no_jerk_time = np.array(err_abs_k_no_jerk.index - err_abs_k_no_jerk.index[0])

    err_abs_x_with_jerk_vals = np.array(err_abs_x_with_jerk.values)
    err_abs_x_with_jerk_time = np.array(err_abs_x_with_jerk.index - err_abs_x_with_jerk.index[0])
    err_abs_y_with_jerk_vals = np.array(err_abs_y_with_jerk.values)
    err_abs_y_with_jerk_time = np.array(err_abs_y_with_jerk.index - err_abs_y_with_jerk.index[0])
    err_abs_z_with_jerk_vals = np.array(err_abs_z_with_jerk.values)
    err_abs_z_with_jerk_time = np.array(err_abs_z_with_jerk.index - err_abs_z_with_jerk.index[0])
    err_abs_k_with_jerk_vals = np.array(err_abs_k_with_jerk.values)
    err_abs_k_with_jerk_time = np.array(err_abs_k_with_jerk.index - err_abs_k_with_jerk.index[0])

    p1 = client.new_plot()
    add_curves_to_kst([err_abs_x_no_jerk_time,err_abs_x_no_jerk_vals,err_abs_x_with_jerk_time,err_abs_x_with_jerk_vals],["time","w_{0,flc}","time","w_{0,fc}"],p1)    
    p2 = client.new_plot()
    add_curves_to_kst([err_abs_y_no_jerk_time,err_abs_y_no_jerk_vals,err_abs_y_with_jerk_time,err_abs_y_with_jerk_vals],["time","w_{1,flc}","time","w_{1,fc}"],p2)
    p3 = client.new_plot()
    add_curves_to_kst([err_abs_z_no_jerk_time,err_abs_z_no_jerk_vals,err_abs_z_with_jerk_time,err_abs_z_with_jerk_vals],["time","w_{2,flc}","time","w_{2,fc}"],p3)
    p4 = client.new_plot()
    add_curves_to_kst([err_abs_k_no_jerk_time,err_abs_k_no_jerk_vals,err_abs_k_with_jerk_time,err_abs_k_with_jerk_vals],["time","w_{3,flc}","time","w_{3,fc}"],p4)

    print("ABS RPM FLC: {:.5f}m".format(rpm_ate(err_abs_x_no_jerk, err_abs_y_no_jerk, err_abs_z_no_jerk, err_abs_k_no_jerk)))
    print("ABS RPM FC: {:.5f}m".format(rpm_ate(err_abs_x_with_jerk, err_abs_y_with_jerk, err_abs_z_with_jerk, err_abs_k_with_jerk)))

def graph_drag(df1, channels):

    drag_x = df1[channels[0]].dropna()    
    drag_y = df1[channels[1]].dropna()
    drag_z = df1[channels[2]].dropna()

    est_drag_x = df1[channels[3]].dropna()    
    est_drag_y = df1[channels[4]].dropna()
    est_drag_z = df1[channels[5]].dropna()

    v_x = df1[channels[9]].dropna()
    v_y = df1[channels[10]].dropna()
    v_z = df1[channels[11]].dropna()

    w_x = df1[channels[12]].dropna()
    w_y = df1[channels[13]].dropna()
    w_z = df1[channels[14]].dropna()

    est_drag_x, drag_x = reindex_down(est_drag_x, drag_x)
    est_drag_y, drag_y = reindex_down(est_drag_y, drag_y)
    est_drag_z, drag_z = reindex_down(est_drag_z, drag_z)

    command_thrust = df1[channels[8]].dropna()

    v_mag = euclidean_magnitude(v_x, v_y, v_z)
    w_mag = euclidean_magnitude(w_x, w_y, w_z)
    
    v_vals = np.array(v_mag.values)
    v_time = np.array(v_mag.index - v_mag.index[0])
    w_vals = np.array(w_mag.values)
    w_time = np.array(w_mag.index - w_mag.index[0])

    #err_abs_mag_with_jerk = euclidean_magnitude(err_abs_x_with_jerk, err_abs_y_with_jerk, err_abs_z_with_jerk)
    #err_abs_mag_no_jerk = euclidean_magnitude(err_abs_x_no_jerk, err_abs_y_no_jerk, err_abs_z_no_jerk)

    drag_x_vals = np.array(drag_x.values)
    drag_x_time = np.array(drag_x.index - drag_x.index[0])
    drag_y_vals = np.array(drag_y.values)
    drag_y_time = np.array(drag_y.index - drag_y.index[0])
    drag_z_vals = np.array(drag_z.values)
    drag_z_time = np.array(drag_z.index - drag_z.index[0])

    est_drag_x_vals = np.array(est_drag_x.values)
    est_drag_x_time = np.array(est_drag_x.index - est_drag_x.index[0])
    est_drag_y_vals = np.array(est_drag_y.values)
    est_drag_y_time = np.array(est_drag_y.index - est_drag_y.index[0])
    est_drag_z_vals = np.array(est_drag_z.values)
    est_drag_z_time = np.array(est_drag_z.index - est_drag_z.index[0])

    command_thrust_vals = np.array(command_thrust.values)
    command_thrust_time = np.array(command_thrust.index - command_thrust.index[0])
    drag_mag = euclidean_magnitude(drag_x, drag_y, drag_z)
    est_drag_mag = euclidean_magnitude(est_drag_x, est_drag_y, est_drag_z)

    drag_mag_vals = np.array(drag_mag.values)
    drag_mag_time = np.array(drag_mag.index - drag_mag.index[0])

    est_drag_mag_vals = np.array(est_drag_mag.values)
    est_drag_mag_time = np.array(est_drag_mag.index - est_drag_mag.index[0])

    p1 = client.new_plot()
    add_curves_to_kst([drag_x_time,drag_x_vals,est_drag_x_time,est_drag_x_vals],["time","F_{d,x}","time","Fe_{d,x}"],p1)    
    
    p2 = client.new_plot()
    add_curves_to_kst([drag_y_time,drag_y_vals,est_drag_y_time,est_drag_y_vals],["time","F_{d,y}","time","Fe_{d,y}"],p2)
    
    p3 = client.new_plot()
    add_curves_to_kst([drag_z_time,drag_z_vals,est_drag_z_time,est_drag_z_vals],["time","F_{d,z}","time","Fe_{d,z}"],p3)
    
    p4 = client.new_plot()
    add_curves_to_kst([command_thrust_time,command_thrust_vals, drag_mag_time, drag_mag_vals],["time","T", "time", "|F_d|"],p4)
    
    p5 = client.new_plot()
    add_curves_to_kst([v_time,v_vals,w_time,w_vals],["time","|v|","time","|\omega|"],p5)
    print("v avg: {} m/s".format(np.mean(v_vals)))

    p6 = client.new_plot()
    add_curves_to_kst([est_drag_mag_time,est_drag_mag_vals, drag_mag_time, drag_mag_vals],["time","|Fe_d|", "time", "|F_d|"],p6)

    print("Rotor Drag Absolute Error: {:.6f} N".format(ate(drag_x - est_drag_x, drag_y - est_drag_y, drag_z - est_drag_z)))
    #print("ATE FC: {:.5f}m".format(ate(err_abs_x_with_jerk, err_abs_y_with_jerk, err_abs_z_with_jerk)))

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    random.seed(args.random)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    client=kst.Client("NumpyVector")

    if len(args.bag) == 1:
        if args.type == 'j':
            channels = ["/riseq/uav/jerk/vector/x", "/riseq/uav/jerk/vector/y", "/riseq/uav/jerk/vector/z", 
                        "/riseq/uav/sensors/imu_corrected/linear_acceleration/x", "/riseq/uav/sensors/imu_corrected/linear_acceleration/y", "/riseq/uav/sensors/imu_corrected/linear_acceleration/z"]
                        #"/vimo_estimator/odometry/twist/twist/linear/x", "/vimo_estimator/odometry/twist/twist/linear/y", "/vimo_estimator/odometry/twist/twist/linear/z",
                        #"/riseq/uav/sensors/imu_corrected/angular_velocity/x", "/riseq/uav/sensors/imu_corrected/angular_velocity/y", "/riseq/uav/sensors/imu_corrected/angular_velocity/z"]
            topics = rosbag_pandas.topics_from_keys(channels)  
            df = rosbag_pandas.bag_to_dataframe(args.bag[0], include=topics)
            print(args.bag[0].split("/")[-1].split(".")[0])
            graph_accelerations2(df, channels)
        if args.type == 'd':
            
            channels = ["/uav/rotordrag/vector/x", "/uav/rotordrag/vector/y", "/uav/rotordrag/vector/z",
                        "/riseq/uav/rotordrag/vector/x", "/riseq/uav/rotordrag/vector/y", "/riseq/uav/rotordrag/vector/z",
                        "/uav/input/rateThrust/thrust/x", "/uav/input/rateThrust/thrust/y", "/uav/input/rateThrust/thrust/z",
                        "/uav/state/twist/twist/linear/x", "/uav/state/twist/twist/linear/y", "/uav/state/twist/twist/linear/z",
                        "/uav/state/twist/twist/angular/x", "/uav/state/twist/twist/angular/y", "/uav/state/twist/twist/angular/z"]
            topics = rosbag_pandas.topics_from_keys(channels)
            df = rosbag_pandas.bag_to_dataframe(args.bag[0], include=topics)
            print(args.bag[0].split("/")[-1].split(".")[0])
            graph_drag(df, channels)
    else:
        if args.type == 'e':
            channels = ["/riseq/uav/error_absolute/vector/x", "/riseq/uav/error_absolute/vector/y", "/riseq/uav/error_absolute/vector/z",
                        "/uav/state/twist/twist/linear/x", "/uav/state/twist/twist/linear/y", "/uav/state/twist/twist/linear/z",
                        "/uav/state/twist/twist/angular/x", "/uav/state/twist/twist/angular/y", "/uav/state/twist/twist/angular/z"]
            topics = rosbag_pandas.topics_from_keys(channels)
            print(0,args.bag[0].split("/")[-1])
            print(1,args.bag[1].split("/")[-1])
            df1 = rosbag_pandas.bag_to_dataframe(args.bag[0], include=topics)
            df2 = rosbag_pandas.bag_to_dataframe(args.bag[1], include=topics)
            graph_errors(df1,df2,channels,args.random)
        if args.type == 'r':
            channels = ["/uav/input/motorspeed/angular_velocities/0", "/uav/input/motorspeed/angular_velocities/1", "/uav/input/motorspeed/angular_velocities/2", "/uav/input/motorspeed/angular_velocities/3"]
            topics = rosbag_pandas.topics_from_keys(channels)
            df1 = rosbag_pandas.bag_to_dataframe(args.bag[0], include=topics)
            df2 = rosbag_pandas.bag_to_dataframe(args.bag[1], include=topics)
            graph_rpms(df1,df2,channels)
