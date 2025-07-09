# -*- coding: utf-8 -*-
"""
Created on Sat May 17 14:30:07 2025

@author: jfr125
"""

 # -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter



def load_clift_data(std_data):
    clift_data = np.loadtxt(std_data, delimiter=",", skiprows=1)
    return clift_data

def load_files(data_set):
    """ load in data from a cvs file cap file data to 100 data points"""
    data = np.loadtxt(data_set, delimiter="\t")
    
    data = data[:100]
    time = np.array(data[:,[0]])
    probe_velocity = np.array(data[:,[1]])
    side_force =np.array(data[:,[2]])
    drag_force = np.array(data[:,[3]])
    lift_force = np.array(data[:,[4]])
    x_moment = np.array(data[:,[5]])
    y_moment = np.array(data[:,[6]])
    z_moment = np.array(data[:,[7]])
    #print("time\n\n",time)
    return time, probe_velocity, side_force, drag_force, lift_force, x_moment, y_moment, z_moment

    
def caculate_Cd(data_sphere, data_string, density, Diameter, Area_tot, Area_string):
    """caculate the drag cofficient"""
    sphere_data = load_files(data_sphere)
    string_data = load_files(data_string)
    
    velocity_sphere = sphere_data[1]
    drag_force_sphere = -1*sphere_data[3]
    
    velocity_string = string_data[1]
    drag_force_string = -1*string_data[3]
    
    Cd_sphere = drag_force_sphere / (0.5 * density * velocity_sphere**2 * Area_tot)
    Cd_string = drag_force_string / (0.5 * density * velocity_sphere**2 * Area_string)
    
    Cd_sphere_average =  np.sum(Cd_sphere)/ np.size(Cd_sphere)
    Cd_string_average =  np.sum(Cd_string)/ np.size(Cd_string)
    
    
    
    # print(drag_force_string.shape)
    #print(velocity_sphere.shape)
    
    #print(drag_force_sphere)
    #print(velocity_sphere)
    #print(density)
    #print(Area_tot)
    #print(Cd_sphere)
    #print(drag_force_string)
    #print(velocity_string)
    #print(Area_string)
    #print(Cd_string)
    
    #print(Cd_string_average)
    #print(Cd_sphere_average)
    return Cd_sphere, Cd_string, velocity_sphere


def caculate_Reynalds_number(data_sphere, data_string, Diameter_sphere, Diameter_string ,kinematic_viscosity):
    """caculate the reynalds number"""
    sphere_data = load_files(data_sphere)
    string_data = load_files(data_string)
    velocity_sphere = sphere_data[1]
    velocity_string = string_data[1]
    
    Re_sphere = (velocity_sphere * Diameter_sphere) / (kinematic_viscosity)
    Re_string = (velocity_string * Diameter_string) / (kinematic_viscosity) 
    
    Re_sphere_average =  np.sum(Re_sphere)/ np.size(Re_sphere)
    Re_string_average =  np.sum(Re_string)/ np.size(Re_string)
    #print(velocity_sphere)
    #print(Diameter_sphere)
    #print(kinematic_viscosity)
    #print(Re_sphere)
    #print(velocity_sphere)
    #print(Diameter_string)
    #print(Re_string)
    
    #print(Re_string_average) 
    #print(Re_sphere_average)
    return Re_sphere, Re_string

def string_drag_corrolation (data_sphere, data_string, density, sphere_frontal_area):
    sphere_data = load_files(data_sphere)
    string_data = load_files(data_string)
    drag_force_sphere = -1*sphere_data[3]
    drag_force_string = -1*string_data[3]
    velocity_sphere = sphere_data[1]
    
    drag_force = drag_force_sphere - drag_force_string
    CDs_rAs_r = (2 * drag_force_sphere)/(density * velocity_sphere ** 2)
    CD_r_A = ( 2 * drag_force_string) / (density * velocity_sphere **2 ) 
    corrected_Cd = (CDs_rAs_r - CD_r_A) / sphere_frontal_area
    #print(drag_force)
    #print(string_drag_correction)
    #print(CD_r_A)
    #print(corrected_Cd)
    return drag_force, CDs_rAs_r, CD_r_A, corrected_Cd 
    
    
def blockage_corrolation (data_sphere, data_string, speed_of_sound, Diameter_sphere, sphere_frontal_area, n_power,  k3s, H_tunnel, B_breadth, cross_sectional_area_tunnel, density):
    sphere_data = load_files(data_sphere)
    string_data = load_files(data_string)
    velocity_sphere = sphere_data[1]
    string_drag  = string_drag_corrolation (data_sphere, data_string, density, sphere_frontal_area)
    corrected_Cd = string_drag[3]
    
    Mach_number = velocity_sphere / speed_of_sound
    
    beta = (1- Mach_number ** 2) ** (1/2)
    
    epleson_3_s = (
        k3s / (beta ** 3) 
        * (B_breadth / H_tunnel + H_tunnel / B_breadth) ** n_power 
        * ((1/6 * np.pi * Diameter_sphere ** 3) / ( 2 * Diameter_sphere))** (1/2) 
        * (sphere_frontal_area) / (cross_sectional_area_tunnel ** (3/2)) )
    
    Cd_sphere_corr_solid_blockage  =  corrected_Cd / ((1 + epleson_3_s) ** 2)
    
    theta_factor = 0.96+1.94 *np.exp(-0.06 * Diameter_sphere / Diameter_sphere)
    
    # Cd_sphere_wake_blockage = (np.sqrt(-1 + (1 + 4 * theta_factor * (sphere_frontal_area / cross_sectional_area_tunnel) * Cd_sphere_corr_solid_blockage))
    #                             / (2 * theta_factor * sphere_frontal_area / cross_sectional_area_tunnel))

    c_d_numerator = -1 + np.sqrt(1 + 4 * theta_factor * (sphere_frontal_area / cross_sectional_area_tunnel) * Cd_sphere_corr_solid_blockage)
    c_d_denominator = 2 * theta_factor * (sphere_frontal_area / cross_sectional_area_tunnel)
    Cd_sphere_wake_blockage = c_d_numerator / c_d_denominator
    
    Cd_sphere_wake_blockage_average =  np.sum(Cd_sphere_wake_blockage )/ (np.size(Cd_sphere_wake_blockage))
    
    #print(Cd_sphere_wake_blockage)
    
    #print(Mach_number)
    #print(beta)
    #print(epleson_3_s)
    #print(Cd_sphere_corr)
    #print(corrected_Cd)
    #print(theta_factor)
    #print(Cd_sphere_wake_blockage)
    
    # print(velocity_sphere)
    # print(Diameter_sphere)
    # print(k3s)
    # print(beta)
    #print((B_breadth))
    #print((H_tunnel))
    #print(n_power)
    #print(Diameter_sphere)
    #print(sphere_frontal_area)
    #print(cross_sectional_area_tunnel)
    return Mach_number, beta, epleson_3_s, Cd_sphere_corr_solid_blockage, theta_factor, Cd_sphere_wake_blockage,  Cd_sphere_wake_blockage_average 
    



def log_plot_Re_Cd(clift_data, density, exp_Re_values=None, exp_Cd_values=None):
    """Plot the Reynolds number and Cd values in log-log scale."""
    Re_clift = np.array(clift_data[:, 0])
    cd_clift = np.array(clift_data[:, 1])

    

    fig, ax = plt.subplots()
    ax.scatter(Re_clift, cd_clift, label="Clift Data", s=3)

    # Add experimental data if provided
    if exp_Re_values is not None and exp_Cd_values is not None:
        Re_50mm_sphere = exp_Re_values[:3]
        Cd_50mm_sphere = exp_Cd_values[:3]
        Re_202mm_sphere = exp_Re_values[3:]
        Cd_202mm_sphere = exp_Cd_values[3:]
        print( Re_50mm_sphere)
        print( Cd_50mm_sphere,"\n")
        print(Re_202mm_sphere)
        print(Cd_202mm_sphere,"\n")
        
        
        ax.scatter(Re_50mm_sphere, Cd_50mm_sphere, color='red', marker='x', label="Experimental results 50mm sphere", s=20, linewidth=0.5)
        ax.scatter(Re_202mm_sphere, Cd_202mm_sphere, color='blue', marker='1', label="Experimental results 202mm sphere", s=25, linewidth=0.5)
        

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid()
    ax.grid(which='minor', linestyle=':', linewidth=0.5)
    ax.set_xlabel("Reynolds Number (Re)")
    ax.set_ylabel("Drag Coefficient (Cd)")

    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='x')

    ax.set_yticks([0.01, 0.1, 1, 10, 100])
    ax.get_yaxis().set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='y')
    ax.legend()
    plt.show()
     
    
def linear_plot_Re_Cd(exp_Re, 
                      experimental_Cd_string_sphere, 
                      experimental_Cd_corrected_string_sphere,
                      experimental_Cd_solid_blockage,
                      experimental_Cd_wake_string_sphere,
                      Cs_smooth_sphere,
                      ave_velocity_list):
    """ plot the reynalds number aganist drag cofficient"""
    
    
    x_ticks = np.linspace(0, 200000, 5)
    y_ticks = np.linspace(0, 1, 11)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # plots for 50mm sphere
    ax1.scatter(exp_Re[:3],  experimental_Cd_string_sphere[:3], color="green", marker="x", label="Measured sphere + string",  s=30, linewidths=0.5)
    ax1.scatter(exp_Re[:3], experimental_Cd_corrected_string_sphere[:3], marker="d", facecolors='none', edgecolors="gold", label="string drag correction", s=30, linewidths=0.5)
    ax1.scatter(exp_Re[:3], experimental_Cd_solid_blockage[:3], marker="*",facecolors='none', edgecolors="magenta", label="Corrected for string drag and solid blockage", s=30, linewidths=0.5)
    ax1.scatter(exp_Re[:3], experimental_Cd_wake_string_sphere[:3], marker="^",facecolors='none', edgecolors="red", label="Corrected for string drag, solid and wake blockage", s=30, linewidths=0.5)
    ax1.scatter(exp_Re[:3], Cs_smooth_sphere[:3], marker=".", facecolors='none', edgecolors="Blue", label="Standard data from clift", s=30, linewidths=0.5)
    
    ax1.set_xticks(x_ticks)
    ax1.set_yticks(y_ticks)
    ax1.minorticks_on() 
    ax1.set_xlabel("Reynolds Number (Re)")
    ax1.set_ylabel("Drag Coefficient (Cd)")
    
    ax1.grid(which='major', linestyle='-', linewidth=0.75)
    ax1.grid(which='minor', linestyle=':', linewidth=0.5)
    ax1.legend(loc="lower right", title=f"50mm sphere with average wind speeds of \n {ave_velocity_list[:3]} m/s" )
    
    # plots for 202mm sphere
    ax2.scatter(exp_Re[3:],  experimental_Cd_string_sphere[3:], color="green", marker="x", label="Measured sphere + string",  s=30, linewidths=0.5)
    ax2.scatter(exp_Re[3:], experimental_Cd_corrected_string_sphere[3:], marker="d", facecolors='none', edgecolors="gold", label="string drag correction", s=30, linewidths=0.5)
    ax2.scatter(exp_Re[3:], experimental_Cd_solid_blockage[3:], marker="*",facecolors='none', edgecolors="magenta", label="Corrected for string drag and solid blockage", s=30, linewidths=0.5)
    ax2.scatter(exp_Re[3:], experimental_Cd_wake_string_sphere[3:], marker="^",facecolors='none', edgecolors="red", label="Corrected for string drag, solid and wake blockage", s=30, linewidths=0.5)
    ax2.scatter(exp_Re[3:], Cs_smooth_sphere[3:], marker=".", facecolors='none', edgecolors="Blue", label="Standard data from clift", s=30, linewidths=0.5)
    
    ax2.set_xticks(x_ticks)
    ax2.set_yticks(y_ticks)
    ax2.minorticks_on() 
    ax2.set_xlabel("Reynolds Number (Re)")
    ax2.set_ylabel("Drag Coefficient (Cd)")
    
    ax2.grid(which='major', linestyle='-', linewidth=0.75)
    ax2.grid(which='minor', linestyle=':', linewidth=0.5)
    ax2.legend(loc="lower right",  title=f"202mm sphere with average wind speeds of \n {ave_velocity_list[3:]} m/s")
    
    
    
    plt.tight_layout()
    plt.show()
    
    

def main():
    """ main fuction call"""
   
    gas_constent = 287.6               # gas constant {JK^-1 kg^-1}
    speed_of_sound = 343.6          # speed of sound in meter per second 
    
    sting_diameter = 0.01 # string diameter in meters
    string_frontal_area = 0.00215 # string frontal area {m^2}
    string_lenght = 0.00215   # string lenght in meters
    
    diameter_50mm = 0.05 # diameter 50mm sphere[m]
    diameter_202mm = 0.202 # diameter 202 sphere [m]
    
    frontal_area_50mm = 0.25 * np.pi* diameter_50mm ** 2    # frontal area in [m^2]
    frontal_area_202mm = 0.25 * np.pi* diameter_202mm ** 2  # frontal area in [m^2]
    
    tot_frontal_area_50mm = frontal_area_50mm + string_frontal_area
    tot_frontal_area_202mm = frontal_area_202mm + string_frontal_area
    
    temperature_conversion = 273.15 # conversion factor to get degrees to kelvin 
    Temperature = 21        # temperature in degrees
    P_air = 101900          #aphmospheric air pressure
    kinematic_viscosity_air = 1.51*10**-5 # kinematic viscosity [m^2/s]
    
    air_density =  (P_air )/(gas_constent*(Temperature + temperature_conversion))
    
    
    
    
    """data for blockage correction"""
    n_power = 1
    k3s = 0.41
    H_tunnel = 0.905 # wind tunnel height [m]
    B_breadth = 1.22 # wind tunnel breadth [m]
    cross_sectional_area = 1.1041 # wind tunnel crossestion area [m^2]
    
    std_data = "emperical_data_clift.csv"
    
    data_set_1 = "50mm_25ms.txt"
    data_set_2 = "50mm_30ms.txt"
    data_set_3 = "50mm_40ms.txt"
    
    data_set_4 = "202mm_7.5ms.txt"
    data_set_5 = "202mm_10.0ms.txt"
    data_set_6 = "202mm_12.5ms.txt"
    
    data_set_7 = "sting_7.5ms.txt"
    data_set_8 = "sting_10.0ms.txt"
    data_set_9 = "sting_12.5ms.txt"
    
    data_set_10 = "sting_25ms.txt"
    data_set_11 = "sting_30ms.txt"
    data_set_12 = "sting_40ms.txt"
    
    files7_5ms = [data_set_4, data_set_7]
    files10ms = [data_set_5, data_set_8]
    files12_5ms = [data_set_6, data_set_9]
    
    files25ms = [data_set_1, data_set_10]
    files30ms = [data_set_2, data_set_11]
    files40ms = [data_set_3, data_set_12]
    
    clift_data = load_clift_data(std_data)
    
    
    # Collect experimental Re and Cd values
    experimental_Re = []
    
    experimental_Cd_string_sphere = []
    experimental_Cd_string_correction = []
    experimental_Cd_solid_blockage = []
    experimental_Cd_wake_string_sphere = []
    
    std_smooth_sphere = []
    ave_velocity_list = []
    
    
    # Define a list of test cases
    test_cases = [
        (files7_5ms, diameter_202mm, frontal_area_202mm, tot_frontal_area_202mm),
        (files10ms, diameter_202mm, frontal_area_202mm, tot_frontal_area_202mm),
        (files12_5ms, diameter_202mm, frontal_area_202mm, tot_frontal_area_202mm),
        (files25ms, diameter_50mm, frontal_area_50mm, tot_frontal_area_50mm),
        (files30ms, diameter_50mm, frontal_area_50mm, tot_frontal_area_50mm),
        (files40ms, diameter_50mm, frontal_area_50mm, tot_frontal_area_50mm),
    ]
    
    for files, diameter, area, total_area in test_cases:
        # drag cofficients
        Cd_sphere_string, Cd_string, velocity_sphere = caculate_Cd(files[0], files[1], air_density, diameter, total_area, string_frontal_area)
        # reynalds numbers 
        Re_sphere_string, _ = caculate_Reynalds_number(files[0], files[1], diameter, sting_diameter, kinematic_viscosity_air)
        # blockage correction
        _, _, _, Cd_sphere_corr_solid_blockage, _, _, cd_tot_wake_block = blockage_corrolation(
            files[0], files[1], speed_of_sound, diameter, area, n_power, k3s,
            H_tunnel, B_breadth, cross_sectional_area, air_density)
         # string drag correction
        string_drag = string_drag_corrolation(files[0], files[1], air_density, area)  # This gives corrected Cd
        Cd_string_drag_correction = string_drag[3]
        
        
        # Take average Re and Cd
        avg_Re = np.mean(Re_sphere_string)
        ave_Cd_string_sphere = np.mean(Cd_sphere_string)
        avg_Cd_string_drag_correction = np.mean(Cd_string_drag_correction)
        avg_Cd_solid_blockage = np.mean(Cd_sphere_corr_solid_blockage)
        avg_Cd = cd_tot_wake_block
        
        Cd_smooth_sphere = (24 / avg_Re) * (1+0.15*avg_Re ** 0.687) + (0.42 / (1+ 42500 * (avg_Re ** -1.16)))
        avg_velocity_sphere = np.sum(velocity_sphere) / np.size(velocity_sphere)
        
        
        #print(avg_Re)
        #print(avg_Cd)
        #print(Cs_smooth_sphere)
       
        experimental_Re.append(avg_Re)
        experimental_Cd_string_sphere.append(ave_Cd_string_sphere)
        experimental_Cd_solid_blockage.append(avg_Cd_solid_blockage)
        experimental_Cd_wake_string_sphere.append(avg_Cd)
        std_smooth_sphere.append(Cd_smooth_sphere) 
        
        experimental_Cd_string_correction.append(avg_Cd_string_drag_correction)
        ave_velocity_list.append(round(avg_velocity_sphere,1))
        
        #print(ave_Cd_string_sphere)
        
    #print(experimental_Re) 
    #print(experimental_Cd_string_sphere)
    #print(experimental_Cd_string_correction)
    #print(experimental_Cd_solid_blockage)
    #print(std_smooth_sphere) 
    # print(ave_velocity_list)
    
    log_plot_Re_Cd(clift_data, air_density, experimental_Re, experimental_Cd_wake_string_sphere)
    
    linear_plot_Re_Cd(experimental_Re, 
                       experimental_Cd_string_sphere,
                       experimental_Cd_string_correction, 
                       experimental_Cd_solid_blockage,
                       experimental_Cd_wake_string_sphere, 
                       std_smooth_sphere,
                       ave_velocity_list)
       

    
main()

