#!/usr/bin/env python

import argparse
import logging

import matplotlib.pyplot as plt
import rosbag_pandas
import numpy as np

import pykst as kst

def build_parser():
    """
    Builds the parser for reading the command line arguments
    :return: Argument parser
    """
    parser = argparse.ArgumentParser(description='Bagfile key to graph')
    parser.add_argument('-b', '--bag', help='Bag file to read',
                        required=True, nargs='*')
    parser.add_argument('-k', '--key',
                        help='Key you would like to plot',
                        required=False, nargs='*')
    parser.add_argument('-y ', '--ylim',
                        help='Set min and max y lim',
                        required=False, nargs=2)
    parser.add_argument('-c', '--combined',
                        help="Graph them all on one",
                        required=False, action="store_true", dest="sharey")
    parser.add_argument('-o', '--offset',
                        help="offsets for each topic",
                        default=None)
    parser.add_argument('-l', '--labels',
                        help='Labels to use instead of topic names, topic names otherwise',
                        default = None, nargs='*')
    parser.add_argument('-s', '--substract',
                        help='Index of keys to substract , ex: -s 0,1 2,4',
                        default = None, nargs = '*')
    parser.add_argument('-t', '--title',
                        help='Title for the figure',
                        default = None, type = str )
    parser.add_argument('-a', '--ylabel',
                        help='Label for vertical axis',
                        default = None, type = str )
    parser.add_argument('-d', '--xlabel',
                        help='Label for horizontal axis',
                        default = None, type = str )
    return parser

def reindex(a,b,c):
    print("a",len(a.index),"b",len(b.index),"c",len(c.index))
    if (len(a.index) > len(b.index)):
        b = b.reindex(index = a.index, method = 'bfill')
        c = c.reindex(index = a.index, method = 'bfill')
        a = a.fillna(method="bfill")
    else:
        a = a.reindex(index = b.index, method = 'bfill')
        c = c.reindex(index = b.index, method = 'bfill')
        b = b.fillna(method="bfill")
    return [a,b,c]

def ate(x,y,z):
    e = x**2 + y**2 + z**2
    e = e.sum()
    e = e/len(x)
    e = np.sqrt(e)
    return e

def euclidean_magnitude(a,b,c):
    return np.sqrt(a**2 + b**2 + c**2)

def graph(df, channels):

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(3,1,1)
    ax2 = fig1.add_subplot(3,1,2)
    ax3 = fig1.add_subplot(3,1,3)
    #x = df.index.tolist()
    err_abs_x = df[channels[0]].dropna()
    err_alt_x = df[channels[1]].dropna()
    t_vel_x = df[channels[2]].dropna()

    err_abs_y = df[channels[3]].dropna()
    err_alt_y = df[channels[4]].dropna()
    t_vel_y = df[channels[5]].dropna()

    err_abs_z = df[channels[6]].dropna()
    err_alt_z = df[channels[7]].dropna()
    t_vel_z = df[channels[8]].dropna()

    err_abs_x, err_alt_x, t_vel_x = reindex(err_abs_x, err_alt_x, t_vel_x)
    err_abs_y, err_alt_y, t_vel_y = reindex(err_abs_y, err_alt_y, t_vel_y)
    err_abs_z, err_alt_z, t_vel_z = reindex(err_abs_z, err_alt_z, t_vel_z)

    
    err_err_x = (err_abs_x - err_alt_x)#.dropna()
    t_vel_x = t_vel_x#.dropna()
    ax1.scatter(np.abs(t_vel_x.values), np.abs(err_err_x.values),s=0.1)
    
    err_err_y = (err_abs_y - err_alt_y)#.dropna()
    t_vel_y = t_vel_y#.dropna()
    ax2.scatter(np.abs(t_vel_y.values), np.abs(err_err_y.values),s=0.1)

    err_err_z = (err_abs_z - err_alt_z)#.dropna()
    t_vel_z = t_vel_z#.dropna()
    ax3.scatter(np.abs(t_vel_z.values), np.abs(err_err_z.values),s=0.1)
    
    print("ATE: {:.5f}m".format(ate(err_abs_x, err_abs_y, err_abs_z)))
    print("ATE2: {:.5f}m".format(ate(err_alt_x, err_alt_y, err_alt_z)))

    #plt.show()

def graph_errors(df1, df2, channels):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(4,1,1)
    ax2 = fig1.add_subplot(4,1,2)
    ax3 = fig1.add_subplot(4,1,3)
    ax4 = fig1.add_subplot(4,1,4)

    err_abs_x_flc = df1[channels[0]].dropna()    
    err_abs_y_flc = df1[channels[3]].dropna()
    err_abs_z_flc = df1[channels[6]].dropna()

    err_abs_x_fc = df2[channels[0]].dropna()    
    err_abs_y_fc = df2[channels[3]].dropna()
    err_abs_z_fc = df2[channels[6]].dropna()

    err_abs_mag_flc = euclidean_magnitude(err_abs_x_flc, err_abs_y_flc, err_abs_z_flc)
    err_abs_mag_fc = euclidean_magnitude(err_abs_x_fc, err_abs_y_fc, err_abs_z_fc)

    ax1.plot(err_abs_x_flc.index - err_abs_x_flc.index[0], err_abs_x_flc.values, label = 'flc: e_x')
    ax2.plot(err_abs_y_flc.index - err_abs_y_flc.index[0], err_abs_y_flc.values, label = 'flc: e_y')
    ax3.plot(err_abs_z_flc.index - err_abs_z_flc.index[0], err_abs_z_flc.values, label = 'flc: e_z')
    ax4.plot(err_abs_mag_flc.index - err_abs_mag_flc.index[0], err_abs_mag_flc.values, label = 'flc: |e|')

    ax1.plot(err_abs_x_fc.index - err_abs_x_fc.index[0], err_abs_x_fc.values, label = 'fc: e_x')
    ax2.plot(err_abs_y_fc.index - err_abs_y_fc.index[0], err_abs_y_fc.values, label = 'fc: e_y')
    ax3.plot(err_abs_z_fc.index - err_abs_z_fc.index[0], err_abs_z_fc.values, label = 'fc: e_z')
    ax4.plot(err_abs_mag_fc.index - err_abs_mag_fc.index[0], err_abs_mag_fc.values, label = 'fc: |e|')

    ax1.legend(loc='best')
    ax2.legend(loc='best')
    ax3.legend(loc='best')
    ax4.legend(loc='best')

def graph_velocities(df1, df2, channels):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(4,1,1)
    ax2 = fig1.add_subplot(4,1,2)
    ax3 = fig1.add_subplot(4,1,3)
    ax4 = fig1.add_subplot(4,1,4)

    t_vel_x_flc = df1[channels[2]].dropna()    
    t_vel_y_flc = df1[channels[5]].dropna()
    t_vel_z_flc = df1[channels[8]].dropna()

    t_vel_x_fc = df2[channels[2]].dropna()    
    t_vel_y_fc = df2[channels[5]].dropna()
    t_vel_z_fc = df2[channels[8]].dropna()

    t_speed_flc = euclidean_magnitude(t_vel_x_flc, t_vel_y_flc, t_vel_z_flc)
    t_speed_fc = euclidean_magnitude(t_vel_x_fc, t_vel_y_fc, t_vel_z_fc)

    ax1.plot(t_vel_x_flc.index - t_vel_x_flc.index[0], t_vel_x_flc.values, label = 'flc: v_x')
    ax2.plot(t_vel_y_flc.index - t_vel_y_flc.index[0], t_vel_y_flc.values, label = 'flc: v_y')
    ax3.plot(t_vel_z_flc.index - t_vel_z_flc.index[0], t_vel_z_flc.values, label = 'flc: v_z')
    ax4.plot(t_speed_flc.index - t_speed_flc.index[0], t_speed_flc.values, label = 'flc: |v|')

    ax1.plot(t_vel_x_fc.index - t_vel_x_fc.index[0], t_vel_x_fc.values, label = 'fc: v_x')
    ax2.plot(t_vel_y_fc.index - t_vel_y_fc.index[0], t_vel_y_fc.values, label = 'fc: v_y')
    ax3.plot(t_vel_z_fc.index - t_vel_z_fc.index[0], t_vel_z_fc.values, label = 'fc: v_z')
    ax4.plot(t_speed_fc.index - t_speed_fc.index[0], t_speed_fc.values, label = 'fc: |v|')

    ax1.legend(loc='best')
    ax2.legend(loc='best')
    ax3.legend(loc='best')
    ax4.legend(loc='best')

    x1 = np.array(t_vel_x_flc.index - t_vel_x_flc.index[0])
    y1 = np.array(t_vel_x_flc.values)
    x2 = np.array(t_vel_x_fc.index - t_vel_x_fc.index[0])
    y2 = np.array(t_vel_x_fc.values)

    a1 = np.array(t_vel_y_flc.index - t_vel_y_flc.index[0])
    b1 = np.array(t_vel_y_flc.values)
    a2 = np.array(t_vel_y_fc.index - t_vel_y_fc.index[0])
    b2 = np.array(t_vel_y_fc.values)

    m1 = np.array(t_vel_z_flc.index - t_vel_z_flc.index[0])
    n1 = np.array(t_vel_z_flc.values)
    m2 = np.array(t_vel_z_fc.index - t_vel_z_fc.index[0])
    n2 = np.array(t_vel_z_fc.values)

    p1 = client.new_plot()
    add_curves_to_kst([x1,y1,x2,y2],["time","\gamma","time","\omega_x"],p1)
    p2 = client.new_plot()
    add_curves_to_kst([a1,b1,a2,b2],["time","\\beta_3","time","\sigma_x"],p2)
    p3 = client.new_plot()
    add_curves_to_kst([m1,n1,m2,n2],["time","\\theta_d","time","\phi_3"],p3)
    """
    # copy the numpy arrays into kst
    V1 = client.new_editable_vector(x1, name="time") # the name is for the label
    V2 = client.new_editable_vector(y1, name="\gamma") # the name is for the label
    V3 = client.new_editable_vector(x2, name="time") # the name is for the label
    V4 = client.new_editable_vector(y2, name="\omega_x") # the name is for the label

    # inside kst, create a curve, a plot, and add the curve to the plot
    c1 = client.new_curve(V1, V2)
    c2 = client.new_curve(V3, V4)
    p1.add(c1)
    p1.add(c2)
    """

def graph_accelerations(df1, channels):


    df1['jx_avg'] = df1[channels[0]]-df1[channels[0]].shift(1).fillna(0)
    df1['jy_avg'] = df1[channels[0]]-df1[channels[0]].shift(1).fillna(0)
    df1['jz_avg'] = df1[channels[0]]-df1[channels[0]].shift(1).fillna(0)
    df1['jx_int'] =  df1['jx_avg'].cumsum()*0.01
    df1['jy_int'] =  df1['jy_avg'].cumsum()*0.01
    df1['jz_int'] =  df1['jz_avg'].cumsum()*0.01


    jx = df1[channels[0]].dropna()
    jy = df1[channels[1]].dropna()
    jz = df1[channels[2]].dropna()

    """
    ax = df1[channels[3]].dropna()
    ay = df1[channels[4]].dropna()
    az = df1[channels[5]].dropna()


    p1 = client.new_plot()
    add_curves_to_kst([x1,y1,x2,y2],["time","\gamma","time","\omega_x"],p1)
    p2 = client.new_plot()
    add_curves_to_kst([a1,b1,a2,b2],["time","\\beta_3","time","\sigma_x"],p2)
    p3 = client.new_plot()
    add_curves_to_kst([m1,n1,m2,n2],["time","\\theta_d","time","\phi_3"],p3)
    """

def add_curves_to_kst(pairs, name_pairs, plot):

    index = np.arange(0,len(pairs),2)

    for i in index:
        x1 = np.array(pairs[i])
        y1 = np.array(pairs[i+1])

        V1 = client.new_editable_vector(x1, name=name_pairs[i]) # the name is for the label
        V2 = client.new_editable_vector(y1, name=name_pairs[i+1]) # the name is for the label  

        c1 = client.new_curve(V1, V2)
        plot.add(c1)


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    channels = ["/riseq/uav/error_absolute/vector/x", "/riseq/uav/error_alternative/vector/x", "/riseq/uav/error_stats/vector/x",
                "/riseq/uav/error_absolute/vector/y", "/riseq/uav/error_alternative/vector/y", "/riseq/uav/error_stats/vector/y",
                "/riseq/uav/error_absolute/vector/z", "/riseq/uav/error_alternative/vector/z", "/riseq/uav/error_stats/vector/z",
                "/uav_ref_trajectory/pose/position/x", "/uav_ref_trajectory/pose/position/y", "/uav_ref_trajectory/pose/position/z"]

    channels = ["/riseq/uav/jerk/vector/x", "/riseq/uav/jerk/vector/y", "/riseq/uav/jerk/vector/z", "/riseq/uav/desired_orientation/pose/position/x", "/riseq/uav/desired_orientation/pose/position/y", "/riseq/uav/desired_orientation/pose/position/z"]


    topics = rosbag_pandas.topics_from_keys(channels)

    client=kst.Client("NumpyVector")

    if len(args.bag) == 1:
        df = rosbag_pandas.bag_to_dataframe(args.bag, include=topics)
        graph_accelerations(df, channels)
    elif len(args.bag) == 2:
        df1 = rosbag_pandas.bag_to_dataframe(args.bag[0], include=topics)
        df2 = rosbag_pandas.bag_to_dataframe(args.bag[1], include=topics)
        graph_velocities(df2, df1, channels)
        graph_errors(df2, df1, channels)
        #plt.show()