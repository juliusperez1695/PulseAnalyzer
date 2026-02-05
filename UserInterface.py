from PulseAnalyzer import *

class PulseAnalyzerUI:
    def Initialize(self):
        pulseanalyzer = PulseAnalyzer()
    
        pw_analysis_menu = Menu("Pulse Analyzer: PULSE WIDTH",
                            {
                                '1': {'text':"Single Pulse Width Test (Single Pulse CSV)", 
                                    'action':pulseanalyzer.Run_SinglePW_Analysis},
                                '2': {'text':"Single Pulse Width Test (Pulse Index, Pulse Train CSV)", 
                                    'action':pulseanalyzer.Run_TrainIdxPW_Analysis},
                                '3': {'text':"Pulse Width Trend Test (Pulse Train CSV)", 
                                    'action':pulseanalyzer.Run_PWTrend_Analysis},
                                'Q': {'text':"Main Menu", 
                                    'action':quit}
                            }, exit_msg = "")
        
        level_analysis_menu = Menu("Pulse Analyzer: PULSE LEVELS",
                            {
                                '1': {'text':"Single Pulse Levels Test (Single Pulse CSV)", 
                                    'action':pulseanalyzer.Run_SingleLevels_Analysis},
                                '2': {'text':"Single Pulse Levels Test (Pulse Index, Pulse Train CSV)", 
                                    'action':pulseanalyzer.Run_TrainIdxLevels_Analysis},
                                '3': {'text':"Pulse Levels Trend Test (Pulse Train CSV)", 
                                    'action':pulseanalyzer.Run_LevelsTrend_Analysis},
                                'Q': {'text':"Main Menu", 
                                    'action':quit}
                            }, exit_msg = "")
        
        edgetime_analysis_menu = Menu("Pulse Analyzer: PULSE RISE/FALL TIMES",
                            {
                                '1': {'text':"Single Pulse Rise/Fall Time Test (Single Pulse CSV)", 
                                    'action':pulseanalyzer.Run_SingleEdgeTimes_Analysis},
                                '2': {'text':"Single Pulse Rise/Fall Time Test (Pulse Index, Pulse Train CSV)", 
                                    'action':pulseanalyzer.Run_TrainIdxEdgeTimes_Analysis},
                                '3': {'text':"Pulse Rise/Fall Time Trend Test (Pulse Train CSV)", 
                                    'action':pulseanalyzer.Run_EdgeTimesTrend_Analysis},
                                'Q': {'text':"Main Menu", 
                                    'action':quit}
                            }, exit_msg = "")
        
        MAIN_MENU = Menu("Welcome!\n\nPulse Analyzer: MAIN MENU",
                            {
                                '1': {'text':"Analyze for Pulse Width", 
                                    'action':pw_analysis_menu.Run},
                                '2': {'text':"Analyze for Pulse Levels", 
                                    'action':level_analysis_menu.Run},
                                '3': {'text':"Analyze for Pulse Rise/Fall Times",
                                    'action':edgetime_analysis_menu.Run},
                                'Q': {'text':"Quit", 
                                    'action':quit}
                            }, exit_msg = "Program Terminated - Goodbye!")
        
        return MAIN_MENU
        
    def RUN(self, prog_menu):
        prog_menu.Run()


class Menu:
    def __init__(self, prompt, options, exit_msg):
        self.prompt = prompt
        self.options = options
        self.exit_msg = exit_msg

    def Display(self):
        print("\n"+self.prompt)
        for key, value in self.options.items():
            print(f"[{key}]   {value['text']}")

    def Get_UserChoice(self):
        while True:
            choice = input("\nEnter your choice: ").upper()
            
            if choice == 'Q':
                print("\n"+self.exit_msg)
                return choice
            elif choice in self.options and choice != 'Q':
                print("\nRunning \""+self.options[choice]['text']+"\"")
                return choice
            else:
                print("Invalid input - try again.")

    def Run_UserChoice(self, choice):
        self.options[choice]['action']()

    def Run(self):
        
        while True:
            self.Display()
            choice = self.Get_UserChoice()
            if choice == 'q' or choice == 'Q':
                break
            else:
                self.Run_UserChoice(choice)

