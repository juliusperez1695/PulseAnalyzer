import math, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

class PulseAnalyzer:
    
    def __init__(self):
        None

    def Import_Data(self, datapath):
        df = pd.read_csv(datapath, header=0, dtype='float64', float_precision='round_trip')
        return np.array(df.iloc[:,0]), np.array(df.iloc[:,1])
    
    def Export_Data(self, x_data, y_data):
        
        # Validate user input before attempting to export
        while True:
            datapath = input("Enter the directory path where you want your results saved (ending with '\\'): ")
            if os.path.exists(datapath) and datapath[-1] == '\\':
                break
            else:
                print("Invalid directory - check that the path exists and ends with '\\'.\n")
        
        while True:
            file_name = input("Enter a name for your exported CSV file (include extension '.csv'): ")
            if file_name[-4:] == '.csv':
                break
            else:
                print("Unable to export - check for proper file extension '.csv' in your file name.\n")

        df = pd.DataFrame()
        df[0] = np.array(x_data)
        df[1] = np.array(y_data)
        df.to_csv(datapath + file_name, index=False, header=False)
    
    def Get_Valid_FilePath(self, prompt):
        while True:
            user_path = input("\n"+prompt)
            if os.path.exists(user_path) and user_path[-4:] == '.csv':
                valid_path = user_path
                break
            else:
                print("Invalid file and/or directory path provided \n\t- Check that the path exists and the '.csv' extension is included.")

        return valid_path
    ''' Analytical Functions '''
    def Get_Single_Pulse(self, pulse_num, t_train_data, V_train_data, rise_edge_idxs, fall_edge_idxs, plot=0):
        ''' Generates data subset for a single pulse within a train of pulses '''
        samp_rate = 1/(t_train_data[1] - t_train_data[0])
        V_rise_edges = V_train_data[rise_edge_idxs] # capture only the rising edge crossings
        t_rise_edges = t_train_data[rise_edge_idxs] # capture only the rising edge crossings

        # get time data for single pulse
        period = t_rise_edges[1] - t_rise_edges[0]
        pulse_start_idx = rise_edge_idxs[pulse_num]-math.floor(samp_rate*0.25*period)
        pulse_end_idx = fall_edge_idxs[pulse_num]+math.floor(samp_rate*0.25*period)
        t_pulse = t_train_data[pulse_start_idx:pulse_end_idx]
        V_pulse = V_train_data[pulse_start_idx:pulse_end_idx]

        # plot results
        if plot==1:
            plt.plot(t_train_data, V_train_data, label="Train Data")
            plt.plot(t_rise_edges, V_rise_edges, 'o', label="Edge Points")
            #plt.plot(t_peaks, V_peaks, 'o', label="Peaks")
            plt.plot(t_pulse, V_pulse, label="Pulse Data")
            plt.legend()
            plt.show(block=False)
        
        return t_pulse, V_pulse
    
    def Compute_Avg_PulseLevels(self, t_data, V_data, numTopSamples, numBaseSamples, plot_results=1):
        ''' Computes a value for the pulse top amplitude and the pulse base amplitude using an averaged set of values '''
        # find the highest 100 values of the pulse
        top_idxs = np.argsort(V_data)[-numTopSamples:]
        top_volts = V_data[top_idxs]
        
        # find the lowest 100 values of the pulse
        bot_idxs = np.argsort(V_data)[0:numBaseSamples-1]
        bot_volts = V_data[bot_idxs]

        # take the average of each to determine the pulse top level and pulse base level
        pulse_top = np.mean(top_volts)
        pulse_base = np.mean(bot_volts)

        if plot_results:
            # plot the results
            plt.figure()
            plt.plot(t_data, V_data, label='$V(t)$')
            plt.axhline(y=pulse_top, linestyle="dotted", color="k",label='$V_{top}$')
            plt.axhline(y=pulse_base, linestyle="--", color="k",label='$V_{base}$')
            plt.title("Averaged Top/Base Amplitude")
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage (V)")
            plt.legend()
            plt.show(block=False)

        return pulse_top, pulse_base

    def Compute_RiseFall_Times(self, t_data, V_data, pulse_top, pulse_base, plot_results=1):
        
        # Do the same edge fitting done in the Pulse Width analysis

        # compute Vthres from pulse_top and pulse_base
        Vthres = (pulse_top + pulse_base) / 2
    
        # Create an initial search list of data points around the rising edge using the absolute max and min of the data
        pulse_max_idx = np.argmax(V_data)
        pulse_min_idx = np.argmin(V_data[0:pulse_max_idx])
        searchlist_rise = V_data[0:pulse_max_idx]
        timelist_rise = t_data[0:pulse_max_idx]
    
        # perform a linear regression fit and optimize R_sq value for rising edge
        m_rise, b_rise, _, _, _ = self._Run_LinearOptimizer(searchlist_rise, timelist_rise, Vthres, rising=1)
    
        # generate line of best fit
        y_line_rise = m_rise*timelist_rise + b_rise

        # Compute the 10 and 90 percent level times on the rising edge
        t_tenperc_r = (1.1*pulse_base - b_rise)/m_rise
        t_ninetyperc_r = (0.9*pulse_top - b_rise)/m_rise
        rise_t = t_ninetyperc_r - t_tenperc_r
        
        # Repeat the same for the falling edge
        searchlist_fall = V_data[pulse_max_idx+1:]
        timelist_fall = t_data[pulse_max_idx+1:]
        m_fall, b_fall, _, _, _ = self._Run_LinearOptimizer(searchlist_fall, timelist_fall, Vthres, rising=0)
        y_line_fall = m_fall*timelist_fall + b_fall

        # Compute the 10 and 90 percent level times on the falling edge
        t_tenperc_f = (1.1*pulse_base - b_fall)/m_fall
        t_ninetyperc_f = (0.9*pulse_top - b_fall)/m_fall
        fall_t = t_tenperc_f - t_ninetyperc_f

        if plot_results:
            # plot the results
            plt.figure()
            plt.plot(t_data, V_data, label='$V(t)$')
            plt.axhline(y=0.9*pulse_top, linestyle="dotted", color="k",label='$V_{90perc}$')
            plt.axhline(y=1.1*pulse_base, linestyle="--", color="k",label='$V_{10perc}$')
            plt.plot(timelist_rise, y_line_rise, 'r', label="Rise Time = "+str(rise_t))
            plt.plot(timelist_fall, y_line_fall, 'b', label="Fall Time = "+str(fall_t))
            plt.title("Rise/Fall Time")
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage (V)")
            plt.xlim(t_data[0], t_data[-1])
            plt.ylim(0.9*V_data[pulse_min_idx],1.1*V_data[pulse_max_idx])
            plt.legend()
            plt.show(block=False)

        return rise_t, fall_t

    def Compute_LinRegr_PW(self, t_data, V_data, pulse_top, pulse_base, plot_results=1):
        ''' Computes the time delta between the rising and falling edges along a voltage threshold '''
        
        # compute Vthres from pulse_top and pulse_base
        Vthres = (pulse_top + pulse_base) / 2
    
        # Create an initial search list of data points around the rising edge using the absolute max and min of the data
        pulse_max_idx = np.argmax(V_data)
        pulse_min_idx = np.argmin(V_data[0:pulse_max_idx])
        searchlist_rise = V_data[0:pulse_max_idx]
        timelist_rise = t_data[0:pulse_max_idx]
    
        # perform a linear regression fit and optimize R_sq value for rising edge
        m_rise, b_rise, t_thres_rise, R_sq_rise, regr_pts_rise = self._Run_LinearOptimizer(searchlist_rise, timelist_rise, Vthres, rising=1)
    
        # generate line of best fit
        y_line_rise = m_rise*timelist_rise + b_rise
        
        # Repeat the same for the falling edge
        searchlist_fall = V_data[pulse_max_idx+1:]
        timelist_fall = t_data[pulse_max_idx+1:]
        m_fall, b_fall, t_thres_fall, R_sq_fall, regr_pts_fall = self._Run_LinearOptimizer(searchlist_fall, timelist_fall, Vthres, rising=0)
        y_line_fall = m_fall*timelist_fall + b_fall
        
        if(plot_results):
            # Report and plot optimized results
            print("\nComputed time at threshold (RISING EDGE): %.6e" % t_thres_rise[0])
            print("Number of points used for best fit: %d" % regr_pts_rise)
            print("R_sq_rise = %.8f" % R_sq_rise)
            print("\nComputed time at threshold (FALLING EDGE): %.6e" % t_thres_fall[0])
            print("Number of points used for best fit: %d" % regr_pts_fall)
            print("R_sq_fall = %.8f" % R_sq_fall)
            
            plt.figure()
            plt.plot(t_data, V_data)
            plt.axhline(y=Vthres, linestyle="--", color="k",label='$V_{thres}$')
            plt.plot(timelist_rise, y_line_rise, 'r', label="Rise Fit")
            plt.plot(timelist_fall, y_line_fall, 'b', label="Fall Fit")
            plt.xlim(t_data[0], t_data[-1])
            plt.ylim(0.9*V_data[pulse_min_idx],1.1*V_data[pulse_max_idx])
            plt.title("Adjusted FWHM Threshold")
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage (V)")
            plt.legend()
            plt.show(block=False)
    
        return t_thres_fall - t_thres_rise
    
    def Get_PW_Trend(self, t_train, V_train):
        # obtain the edge idxs for isolating the i-th pulse
        rise_edge_idxs, fall_edge_idxs = self._find_edge_idxs(V_train)
        num_pulses = np.min([np.size(rise_edge_idxs),np.size(fall_edge_idxs)]) # may be useful if a pulse is cutoff on either end of the train
        
        # empty lists for storing results for each pulse
        pulse_num = []
        pws = []
        
        plt.figure()
        t_offset = 0 # time offset which will updated for each pulse to be overlaid on top of one another
        for i in range(0,num_pulses):
            # Extract the i-th pulse to be analyzed
            t_pulse, V_pulse = self.Get_Single_Pulse(i, t_train, V_train, rise_edge_idxs, fall_edge_idxs, plot=0)
            
            # For determining the pulse top amplitude, take the range between pulse edges and average the upper percentage
            pts_btwn_pulseEdges = fall_edge_idxs[i] - rise_edge_idxs[i]
            numTopSamples = int(0.6*pts_btwn_pulseEdges)

            # For determining the pulse base amplitude, take entire range of points for the pulse and average the lower percentage
            pts_per_pulse = np.size(t_pulse)
            numBaseSamples = int(0.3*pts_per_pulse)
            
            # Compute the results for pulse top, base, and width
            pulse_top, pulse_base = self.Compute_Avg_PulseLevels(t_pulse, V_pulse, numTopSamples, numBaseSamples, plot_results=0)
            pw = self.Compute_LinRegr_PW(t_pulse, V_pulse, pulse_top, pulse_base, plot_results=0)
            pws.append(pw)
            pulse_num.append(i)

            # For overlaying the pulses, apply a time offset to all pulses following the first one
            if i > 0:
                period = t_train[rise_edge_idxs[i]] - t_train[rise_edge_idxs[i-1]]
                t_offset += period
            plt.plot(t_pulse-t_offset, V_pulse, 'o-', markersize=5)
            print("Pulse #" + str(i+1) + "/" + str(num_pulses) + "; PW = " + str(pw))
        plt.title("Overlaid Train Pulses")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.show(block=False)
        
        return np.array(pulse_num), np.array(pws)

    def Get_Level_Trends(self, t_train, V_train):
        # See Get_PW_Trends() for comment-based help
        
        rise_edge_idxs, fall_edge_idxs = self._find_edge_idxs(V_train)
        num_pulses = np.min([np.size(rise_edge_idxs),np.size(fall_edge_idxs)])
        pulse_num = []
        tops = []
        bases = []

        plt.figure()
        t_offset = 0
        for i in range(0,num_pulses):
            t_pulse, V_pulse = self.Get_Single_Pulse(i, t_train, V_train, rise_edge_idxs, fall_edge_idxs, plot=0)
            pts_btwn_pulseEdges = fall_edge_idxs[i] - rise_edge_idxs[i]
            numTopSamples = int(0.6*pts_btwn_pulseEdges)
            pts_per_pulse = np.size(t_pulse)
            numBaseSamples = int(0.3*pts_per_pulse)
            top, base = self.Compute_Avg_PulseLevels(t_pulse, V_pulse, numTopSamples, numBaseSamples, plot_results=0)
            tops.append(top)
            bases.append(base)
            pulse_num.append(i)

            if i > 0:
                period = t_train[rise_edge_idxs[i]] - t_train[rise_edge_idxs[i-1]]
                t_offset += period
            plt.plot(t_pulse-t_offset, V_pulse, 'o-', markersize=5)
            print("Pulse #" + str(i+1) + "/" + str(num_pulses) + "; Pulse Top Amp. = " + str(top) + "; Pulse Base Amp. = " + str(base))

        plt.title("Overlaid Train Pulses")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.show(block=False)
        
        return np.array(pulse_num), np.array(tops), np.array(bases)
    
    def Get_EdgeTime_Trends(self, t_train, V_train):
        # obtain the edge idxs for isolating the i-th pulse
        rise_edge_idxs, fall_edge_idxs = self._find_edge_idxs(V_train)
        num_pulses = np.min([np.size(rise_edge_idxs),np.size(fall_edge_idxs)]) # may be useful if a pulse is cutoff on either end of the train
        
        # empty lists for storing results for each pulse
        pulse_num = []
        rise_times = []
        fall_times = []
        
        plt.figure()
        t_offset = 0 # time offset which will updated for each pulse to be overlaid on top of one another
        for i in range(0,num_pulses):
            # Extract the i-th pulse to be analyzed
            t_pulse, V_pulse = self.Get_Single_Pulse(i, t_train, V_train, rise_edge_idxs, fall_edge_idxs, plot=0)
            
            # For determining the pulse top amplitude, take the range between pulse edges and average the upper percentage
            pts_btwn_pulseEdges = fall_edge_idxs[i] - rise_edge_idxs[i]
            numTopSamples = int(0.6*pts_btwn_pulseEdges)

            # For determining the pulse base amplitude, take entire range of points for the pulse and average the lower percentage
            pts_per_pulse = np.size(t_pulse)
            numBaseSamples = int(0.3*pts_per_pulse)
            
            # Compute the results for pulse top, base, and width
            pulse_top, pulse_base = self.Compute_Avg_PulseLevels(t_pulse, V_pulse, numTopSamples, numBaseSamples, plot_results=0)
            rise_t, fall_t = self.Compute_RiseFall_Times(t_pulse, V_pulse, pulse_top, pulse_base, plot_results=0)
            rise_times.append(rise_t)
            fall_times.append(fall_t)
            pulse_num.append(i)

            # For overlaying the pulses, apply a time offset to all pulses following the first one
            if i > 0:
                period = t_train[rise_edge_idxs[i]] - t_train[rise_edge_idxs[i-1]]
                t_offset += period
            plt.plot(t_pulse-t_offset, V_pulse, 'o-', markersize=5)
            print("Pulse #" + str(i+1) + "/" + str(num_pulses) + "; Rise Time = " + str(rise_t) + "; Fall Time = " + str(fall_t))
        plt.title("Overlaid Train Pulses")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.show(block=False)
        
        return np.array(pulse_num), np.array(rise_times), np.array(fall_times)
    
    ''' Program RUN Functions '''
    def Run_SinglePW_Analysis(self):
        datapath = self.Get_Valid_FilePath("Enter your single pulse data CSV filepath: ")
        t_data, V_data = self.Import_Data(datapath)
        rise_edge_idxs, fall_edge_idxs = self._find_edge_idxs(V_data)
        pts_btwn_pulseEdges = fall_edge_idxs[0] - rise_edge_idxs[0]
        numTopSamples = int(0.6*pts_btwn_pulseEdges)
        pts_per_pulse = np.size(t_data)
        numBaseSamples = int(0.3*pts_per_pulse)
        pulse_top, pulse_base = self.Compute_Avg_PulseLevels(t_data, V_data, numTopSamples, numBaseSamples, plot_results=1)
        pw = self.Compute_LinRegr_PW(t_data, V_data, pulse_top, pulse_base, plot_results=1)

        print("\nComputed PW = %.6e" % pw[0])

    def Run_TrainIdxPW_Analysis(self):
        datapath = self.Get_Valid_FilePath("Enter your train data CSV filepath: ")
        t_data, V_data = self.Import_Data(datapath)
        rise_edge_idxs, fall_edge_idxs = self._find_edge_idxs(V_data)
        print(str(np.size(rise_edge_idxs)) + " pulses found!\n")

        pulse_num = int(input("Enter the zero-based index of the pulse you want to analyze: "))
        t_pulse, V_pulse = self.Get_Single_Pulse(pulse_num, t_data, V_data, rise_edge_idxs, fall_edge_idxs, plot=0)
        pts_btwn_pulseEdges = fall_edge_idxs[0] - rise_edge_idxs[0]
        numTopSamples = int(0.6*pts_btwn_pulseEdges)
        pts_per_pulse = np.size(t_pulse)
        numBaseSamples = int(0.3*pts_per_pulse)
        pulse_top, pulse_base = self.Compute_Avg_PulseLevels(t_pulse, V_pulse, numTopSamples, numBaseSamples, plot_results=1)
        pw = self.Compute_LinRegr_PW(t_pulse, V_pulse, pulse_top, pulse_base, plot_results=1)

        print("\nComputed PW = %.6e" % pw[0])

    def Run_PWTrend_Analysis(self):
        datapath = self.Get_Valid_FilePath("Enter your train data CSV filepath: ")
        t_data, V_data = self.Import_Data(datapath)
        pulse_num, pws = self.Get_PW_Trend(t_data, V_data)

        while True:
            export_results = input("Export Results? (Y/N): ")
            if(export_results.lower()=='y'):
                do_export = 1
                break
            elif(export_results.lower()=='n'):
                do_export = 0
                break
            else:
                print("Invalid input - try again")
                
        if do_export: 
            self.Export_Data(pulse_num, pws)
        plt.figure()
        plt.plot(pulse_num, pws*1e9, 'o', markersize=3)
        plt.title("Computed PW Trend")
        plt.xlabel("Pulse Index")
        plt.ylabel("Pulse Width (ns)")
        plt.show(block=False)

    def Run_SingleLevels_Analysis(self):
        datapath = self.Get_Valid_FilePath("Enter your single pulse data CSV filepath: ")
        t_data, V_data = self.Import_Data(datapath)
        rise_edge_idxs, fall_edge_idxs = self._find_edge_idxs(V_data)
        pts_btwn_pulseEdges = fall_edge_idxs[0] - rise_edge_idxs[0]
        numTopSamples = int(0.6*pts_btwn_pulseEdges)
        pts_per_pulse = np.size(t_data)
        numBaseSamples = int(0.3*pts_per_pulse)
        pulse_top, pulse_base = self.Compute_Avg_PulseLevels(t_data, V_data, numTopSamples, numBaseSamples, plot_results=1)

        print("\nComputed pulse top = %.6e" % pulse_top)
        print("Computed pulse base = %.6e" % pulse_base)

    def Run_TrainIdxLevels_Analysis(self):
        datapath = self.Get_Valid_FilePath("Enter your train data CSV filepath: ")
        t_data, V_data = self.Import_Data(datapath)
        rise_edge_idxs, fall_edge_idxs = self._find_edge_idxs(V_data)
        print(str(np.size(rise_edge_idxs)) + " pulses found!\n")

        pulse_num = int(input("Enter the zero-based index of the pulse you want to analyze: "))
        t_pulse, V_pulse = self.Get_Single_Pulse(pulse_num, t_data, V_data, rise_edge_idxs, fall_edge_idxs, plot=0)
        pts_btwn_pulseEdges = fall_edge_idxs[0] - rise_edge_idxs[0]
        numTopSamples = int(0.6*pts_btwn_pulseEdges)
        pts_per_pulse = np.size(t_pulse)
        numBaseSamples = int(0.3*pts_per_pulse)
        pulse_top, pulse_base = self.Compute_Avg_PulseLevels(t_pulse, V_pulse, numTopSamples, numBaseSamples, plot_results=1)
        
        print("\nComputed pulse top = %.6e" % pulse_top)
        print("Computed pulse base = %.6e" % pulse_base)

    def Run_LevelsTrend_Analysis(self):
        datapath = self.Get_Valid_FilePath("Enter your train data CSV filepath: ")
        t_data, V_data = self.Import_Data(datapath)
        pulse_num, tops, bases = self.Get_Level_Trends(t_data, V_data)

        while True:
            export_results = input("Export Results? (Y/N): ")
            if(export_results.lower()=='y'):
                do_export = 1
                break
            elif(export_results.lower()=='n'):
                do_export = 0
                break
            else:
                print("Invalid input - try again")
                
        if do_export: 
            print("\nFor Pulse Top results: ")
            self.Export_Data(pulse_num, tops)
            print("\nFor Pulse Base results: ")
            self.Export_Data(pulse_num, bases)

        plt.figure()
        plt.plot(pulse_num, tops, 'o', markersize=3)
        plt.title("Computed Pulse Top Trend")
        plt.xlabel("Pulse Index")
        plt.ylabel("Pulse Top (V)")
        plt.figure()
        plt.plot(pulse_num, bases, '*', markersize=3)
        plt.title("Computed Pulse Base Trend")
        plt.xlabel("Pulse Index")
        plt.ylabel("Pulse Base (V)")
        plt.show(block=False)

    def Run_SingleEdgeTimes_Analysis(self):
        datapath = self.Get_Valid_FilePath("Enter your single pulse data CSV filepath: ")
        t_data, V_data = self.Import_Data(datapath)
        rise_edge_idxs, fall_edge_idxs = self._find_edge_idxs(V_data)
        pts_btwn_pulseEdges = fall_edge_idxs[0] - rise_edge_idxs[0]
        numTopSamples = int(0.6*pts_btwn_pulseEdges)
        pts_per_pulse = np.size(t_data)
        numBaseSamples = int(0.3*pts_per_pulse)
        pulse_top, pulse_base = self.Compute_Avg_PulseLevels(t_data, V_data, numTopSamples, numBaseSamples, plot_results=1)
        rise_t, fall_t = self.Compute_RiseFall_Times(t_data, V_data, pulse_top, pulse_base, plot_results=1)

        print("\nComputed Rise Time = %.6e" % rise_t)
        print("Computed Fall Time = %.6e" % fall_t)

    def Run_TrainIdxEdgeTimes_Analysis(self):
        datapath = self.Get_Valid_FilePath("Enter your train data CSV filepath: ")
        t_data, V_data = self.Import_Data(datapath)
        rise_edge_idxs, fall_edge_idxs = self._find_edge_idxs(V_data)
        print(str(np.size(rise_edge_idxs)) + " pulses found!\n")

        pulse_num = int(input("Enter the zero-based index of the pulse you want to analyze: "))
        t_pulse, V_pulse = self.Get_Single_Pulse(pulse_num, t_data, V_data, rise_edge_idxs, fall_edge_idxs, plot=0)
        pts_btwn_pulseEdges = fall_edge_idxs[0] - rise_edge_idxs[0]
        numTopSamples = int(0.6*pts_btwn_pulseEdges)
        pts_per_pulse = np.size(t_pulse)
        numBaseSamples = int(0.3*pts_per_pulse)
        pulse_top, pulse_base = self.Compute_Avg_PulseLevels(t_pulse, V_pulse, numTopSamples, numBaseSamples, plot_results=1)
        rise_t, fall_t = self.Compute_RiseFall_Times(t_pulse, V_pulse, pulse_top, pulse_base, plot_results=1)

        print("\nComputed Rise Time = %.6e" % rise_t)
        print("Computed Fall Time = %.6e" % fall_t)

    def Run_EdgeTimesTrend_Analysis(self):
        datapath = self.Get_Valid_FilePath("Enter your train data CSV filepath: ")
        t_data, V_data = self.Import_Data(datapath)
        pulse_num, rise_times, fall_times = self.Get_EdgeTime_Trends(t_data, V_data)

        while True:
            export_results = input("Export Results? (Y/N): ")
            if(export_results.lower()=='y'):
                do_export = 1
                break
            elif(export_results.lower()=='n'):
                do_export = 0
                break
            else:
                print("Invalid input - try again")
                
        if do_export: 
            print("For exporting RISE time data: ")
            self.Export_Data(pulse_num, rise_times)
            print("For exporting FALL time data: ")
            self.Export_Data(pulse_num, fall_times)
            
        plt.figure()
        plt.plot(pulse_num, rise_times*1e12, 'o', markersize=3)
        plt.title("Computed Rise Time Trend")
        plt.xlabel("Pulse Index")
        plt.ylabel("Pulse Rise Time (ps)")
        plt.show(block=False)
        plt.figure()
        plt.plot(pulse_num, fall_times*1e12, 'o', markersize=3)
        plt.title("Computed Fall Time Trend")
        plt.xlabel("Pulse Index")
        plt.ylabel("Pulse Fall Time (ps)")
        plt.show(block=False)

    ''' Internal Functions '''
    def _find_edge_idxs(self, V_train):
        ''' Found method online: find the points at which the train data crosses a certain value on the pulse edges '''
        target_val = (np.max(V_train) + np.min(V_train))/2
        
        data = np.asarray(V_train)
        signs = np.sign(data - target_val)
        
        # Find indices where the sign changes
        change_indices = np.where(np.diff(signs))[0] + 1
        rise_edge_idxs = change_indices[::2]
        fall_edge_idxs = change_indices[1::2]

        return rise_edge_idxs, fall_edge_idxs
    
    # Finds the closest n values in an array to the provided target value
    def _find_closest_values(self, arr, target, n=2):
        arr = np.asarray(arr)
        idx = np.sort(np.abs(arr - target).argsort()[:n])
    
        return idx, arr[idx]

    # Generates a linear fit for the rising/falling edge
    def _Get_Edge_Fit(self, subset2080_t, subset2080_V, tenperc_V, ninetyperc_V):
        model = LinearRegression()
        t_pred = model.fit(subset2080_V.reshape((-1,1)), subset2080_t)
        m = 1 / t_pred.coef_[0]
        b = -t_pred.intercept_ / t_pred.coef_[0]
        tenperc_t = model.predict(tenperc_V.reshape((1,-1)))
        ninetyperc_t = model.predict(ninetyperc_V.reshape((1,-1)))

        return m, b, tenperc_t, ninetyperc_t
    
    # Generates a linear model about a given voltage level based on the provided V(t) data
    def _Get_Threshold_Fit(self, subset_t, subset_V, Vthres):
        model = LinearRegression()
        t_pred = model.fit(subset_V.reshape((-1,1)), subset_t)
        m = 1 / t_pred.coef_[0]
        b = -t_pred.intercept_ / t_pred.coef_[0]
        R_sq = model.score(subset_V.reshape((-1,1)), subset_t)
        t_thres = model.predict(Vthres.reshape((1,-1)))
    
        return m, b, t_thres, R_sq

    # Varies the number of points considered near the Voltage threshold when generating a linear model
    def _Run_LinearOptimizer(self, searchlist, timelist, Vthres, rising, print_results=0):
        # Goal is to optimize R_sq value of the linear fit given a max range of samples to test against (see hard-coded for-loop below)
        m_set = []
        b_set = []
        t_thres_set = []
        R_sq_set = []
        
        for num in range(3,10):
            
            subset_idxs, subset_levels = self._find_closest_values(searchlist, Vthres, n=num)
            subset_times = timelist[subset_idxs]
    
            if(print_results):
                if(rising):
                    print("Fitting against %d points on the RISING EDGE" % num)
                    print("**VERIFY POINTS LIE ON THE RISING EDGE**")
                    print("Subset of TIME for linear regression at RISING EDGE")
                    print(subset_times)
                    print("Subset of VOLTAGE for linear regression at RISING EDGE")
                    print(str(subset_levels) + "\n")
                else:
                    print("Fitting against %d points on the FALLING EDGE" % num)
                    print("**VERIFY POINTS LIE ON THE FALLING EDGE**")
                    print("Subset of TIME for linear regression at FALLING EDGE")
                    print(subset_times)
                    print("Subset of VOLTAGE for linear regression at FALLING EDGE")
                    print(str(subset_levels) + "\n")
        
            # generate set of values for each parameter
            m_samp, b_samp, t_thres_samp, R_sq_samp = self._Get_Threshold_Fit(subset_times, subset_levels, Vthres)
            
            m_set.append(m_samp)
            b_set.append(b_samp)
            t_thres_set.append(t_thres_samp)
            R_sq_set.append(R_sq_samp)
    
        R_sq_idx = np.argmax(R_sq_set)
        R_sq = R_sq_set[R_sq_idx]
        m = m_set[R_sq_idx]
        b = b_set[R_sq_idx]
        t_thres = t_thres_set[R_sq_idx]
        num_regr_pts = R_sq_idx + 3 #index offset since range of tested values starts at 3
        
        # return the optimal R_sq and associated values
        return m, b, t_thres, R_sq, num_regr_pts