# This will generate CSV file i.e Data set with setpoints and simulated tclab 
# including all libraries..............

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import time

# For TCLab
import tclab

speedup=100 #70 # this means 70 times faster i.e instead of 1 sec it will run in 0.014 secs
                      # i.e it will finish 5400 cycles in 78 seconds instead of 5400 secs when speedup=1

TCLab = tclab.setup(connected= False, speedup=speedup) # speedup is to speedup compile

# PID Parameters
Kc   = 6.0
tauI = 75.0 # sec
tauD = 0.0  # sec


#__________________ PID Controller_________________________________


def pid(sp,pv,pv_last,ierr,dt):
    # PID coefficients in terms of tuning parameters
    KP = Kc
    KI = Kc / tauI
    KD = Kc * tauD
    
    # ubias for controller (initial heater)
    op0 = 0 
    
    # upper and lower bounds on heater level
    ophi = 100
    oplo = 0
    
    # calculate the error
    error = sp - pv
    
    # calculate the integral error
    ierr = ierr + KI * error * dt
    
    # calculate the measurement derivative
    if dt>=1e-8:
        dpv = (pv - pv_last) / dt
    else:
        dpv = 0.0;
    
    # calculate the PID output
    P = KP * error
    I = ierr
    D = -KD * dpv
    op = op0 + P + I + D
    
    # implement anti-reset windup
    if op < oplo or op > ophi:
        I = I - KI * error * dt
        # clip output
        op = max(oplo,min(ophi,op))
        
    # return the controller output and PID terms
    return [op,P,I,D]


#_______Generate random Set point of a definite range and non-uniform duration_____

# Run time in minutes
run_time = 90    #90 minutes   # 90x60 = 5400 rows  in csv file

# Number of cycles
loops = int(60.0*run_time) # this will give 90x60=5400 rows 

# arrays for storing data
T1 = np.zeros(loops) # measured T (degC)
Q1 = np.zeros(loops) # Heater values
tm = np.zeros(loops) # Time

# Temperature set point (degC)
with TCLab() as lab:
    Tsp1 = np.ones(loops) * lab.T1

# vary temperature setpoint
end = 30 # leave 1st 30 seconds of temp set point as room temp
while end <= loops:
    start = end
    # keep new temp set point value for anywhere from 4 to 10 min
    dur_min=240  # setpoint interval of 4 minutes minimum
    dur_max=600  # setpoint interval of 10 minutes  maximum
    end += random.randint(dur_min,dur_max)
    range_min=30     # amplitude minimum
    range_max=70     # amplitude maximum
    Tsp1[start:end] = random.randint(range_min,range_max)

print('\nrandom set-point of range' ,range_min ,'to' ,range_max, 'with interval of', dur_min, 'to', dur_max,'seconds is Generated.\n')

#____________Plot the randomly generated set point_________________________
plt.plot(Tsp1)
plt.xlabel('Time',size=14)
plt.ylabel(r'Temp SP ($^oC$)',size=14)
plt.xticks(size=12)
plt.yticks(size=12)
plt.savefig('Set-point_profile.png');

# Data collection
with TCLab() as lab:
    # Find current T1, T2
    print('Temperature 1: {0:0.2f} °C'.format(lab.T1))
    print('Temperature 2: {0:0.2f} °C'.format(lab.T2))
    print('PID simulation with random set-points begins\n')

    
    # Integral error
    ierr = 0.0
    # Integral absolute error
    iae = 0.0
    
    prev_time = 0
    for i,t in enumerate(tclab.clock(loops-1)):
        tm[i] = t
        dt = t - prev_time
        
        # Read temperatures in Kelvin 
        T1[i] = lab.T1

        # Integral absolute error
        iae += np.abs(Tsp1[i]-T1[i])

        # Calculate PID output
        [Q1[i],P,ierr,D] = pid(Tsp1[i],T1[i],T1[i-1],ierr,dt)
        
        # Calculate MPC output (optionally replace PID)

        # Write heater output (0-100)
        lab.Q1(Q1[i])

        # Print line of data
        if i%100==0:
            print(('{:6.1f} {:6.2f} {:6.2f} ' + \
                  '{:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f}').format( \
                      tm[i],Tsp1[i],T1[i], \
                      Q1[i],P,ierr,D,iae))
        prev_time = t

#______Save the dataset in csv file_______________________
df = pd.DataFrame()
df['Q1'] = Q1[:i]
df['Q1'].fillna(0,inplace=True)
df['T1'] = T1[:i]
df['Tsp'] = Tsp1[:i]
df['err'] = df['Tsp'] - df['T1'] # this part is newly introduced
df.to_csv('PID_train_data_kc6_tauI75.csv',index=False)
print('\nCSV file containing Q1, T1, Tsp, err is generated')

# Plot
plt.plot(df['Q1'],'b-',label='$Q_1$ (%)')
plt.plot(df['T1'],'r-',label='$T_1$ $(^oC)$')
plt.plot(df['Tsp'],'k-',label='SP $(^oC)$')
plt.legend(loc='upper center')
plt.title(label='PID Simulation waveform', color='g',loc='center')
plt.savefig('PID_train_kc6_tauI75.png');

