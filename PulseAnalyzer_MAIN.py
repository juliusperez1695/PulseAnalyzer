from UserInterface import PulseAnalyzerUI
    
def main():
    # Generate User Interface for the Pulse Analyzer
    prog_UI = PulseAnalyzerUI().Initialize()

    # Execute program
    PulseAnalyzerUI().RUN(prog_UI)

if __name__ == '__main__':
    main()