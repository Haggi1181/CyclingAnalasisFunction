from turtle import distance
import scipy as sp
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.patheffects as path_effects
from statistics import mean
import datetime
import fluids
import glob
import os

plt.close("all")

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199


def ReadData(DataPath):
    """
    Function to Read Data from files downloaded from strava using the source plugin, fixes issues with spaces in names and empty watt values

    Parameters
    ----------
    DataPath : String
        File path for data to read

    Returns
    ---------- 
    Data : Pandas dataframe 
        data located at DataPath
    """


    Data = pd.read_csv(DataPath).fillna(0)

    #fixes the missing values for watts
    Data.columns = Data.columns.str.lstrip()
    Data["watts"] = Data["watts"].replace("      ", int(0))
    Data["watts"] = pd.to_numeric(Data["watts"])

    return Data

def PowerProfileCalculator(DataPath, ScaleFactor=10, InitialSample = 5, output = False, Visualize = True, GrowthModel = "Custom", CustomTimes = [5, 10, 30, 60, 300, 600, 1200, 1800, 3600], PlotName = "nan", leg = False):
    """
    Internal function that calculates the CDA of a bike rider for CDAPlot

    Parameters
    ----------
    DataPath : String
        Path for raw data, file must have a power data
    ScaleFactor : Int
        Number of iterations to perform the growth model for Exp and Lin
    InitialSample : Int
        Start point to calculate power / s
    output : Boolean
        Toggles returns for function
    Visualize : Boolean
        Toggles plotting for function
    GrowthModel : String
        Changes model for finding different scales, 
        options:
            Exp - Doubles each time
            Lin - Adds ScaleFactor each time
            MaxLin - Same as Lin but automatically stops once the largest number of additions have been performed
            MaxEXP - Same as Exp but automatically stops once the largest number of doubles have been performed
            Custom - Records for each int in CustomTimes
    CustomTimes : 1DArray[Int]
        Array of all the time values for use in the Custom option
    PlotName : String
        Passes the temperature recorded, used if TempCorrection = True


    Returns
    ----------
    TimeScales : 1DArray[Float]
        Array of all the x values / lengths sampled
    MaxPowers : 1DArray[Float]
        Array of all the y values / max powers for TimeScales lengths
    """
    RawData = ReadData(DataPath)

    PowerResults = []
    MaxPowers = []
    TimeScales = []
    i=0
    if GrowthModel == "Exp":
        while i < ScaleFactor:
            temp = RawData["watts"].rolling(InitialSample).mean()
            temp = [x for x in temp if np.isnan(x) == False]
            PowerResults.append(temp)
            TimeScales.append(InitialSample)
            InitialSample = InitialSample*2
            i = i + 1
        i=0
    if GrowthModel == "Lin":
        sample = InitialSample
        while i < ScaleFactor:
            temp = RawData["watts"].rolling(sample).mean()
            temp = [x for x in temp if np.isnan(x) == False]
            PowerResults.append(temp)
            TimeScales.append(sample)
            sample = sample + InitialSample
            i = i + 1
        i=0
    if GrowthModel == "MaxLin":
        sample = InitialSample
        while i < len(RawData["watts"]):
            temp = RawData["watts"].rolling(sample).mean()
            temp = [x for x in temp if np.isnan(x) == False]
            PowerResults.append(temp)
            TimeScales.append(sample)
            sample = sample + InitialSample
            i = sample
        i=0
        ScaleFactor = len(PowerResults)
    if GrowthModel == "MaxExp":
        while i < len(RawData["watts"]):
            temp = RawData["watts"].rolling(InitialSample).mean()
            temp = [x for x in temp if np.isnan(x) == False]
            PowerResults.append(temp)
            TimeScales.append(InitialSample)
            InitialSample = InitialSample*2
            i = i + InitialSample
        i=0
        ScaleFactor = len(PowerResults)
    #print(MaxPowers)
    if GrowthModel == "Custom":
        while i < len(CustomTimes):
            temp = RawData["watts"].rolling(CustomTimes[i]).mean()
            temp = [x for x in temp if np.isnan(x) == False]
            PowerResults.append(temp)
            TimeScales.append(CustomTimes[i])
            i = i + 1
        i=0
        ScaleFactor = len(CustomTimes)
    while i < ScaleFactor:
        #print(i)
        #print(max(PowerResults[i]))
        MaxPowers.append(max(PowerResults[i]))
        i = i + 1
    if Visualize == True:
        plt.xlabel("Time / s")
        plt.ylabel("Power / w")
        plt.plot(TimeScales, MaxPowers, label = PlotName)
        if leg == True:
            plt.legend()

    if output == True:
        return(TimeScales,MaxPowers)


def FolderPowerProfileCalculator(FolderPath, PrintRawData = False, GrowthModel = "MaxExp", CustomTimes = [5, 10, 30, 60, 300, 600, 1200, 1800, 3600], ScaleFactor=10, InitialSample = 5):
    """
    Internal function that calculates the CDA of a bike rider for CDAPlot

    Parameters
    ----------
    FolderPath : String
        Path for folder containing raw data, files must have a power data, must be no other .txt files in that directory
    PrintRawData : Boolean
        Toggles outputting raw data as a printed output
    GrowthModel : String
        Changes model for finding different scales, 
        options:
            Exp - Doubles each time
            Lin - Adds each time
            Max - Increases by 1 each time (warning runs slow, for longer rides large scale processing using this becomes impractical)
            MaxEXP - Same as Exp but automatically stops once the largest number of doubles have been performed
            Custom - Records for each int in CustomTimes
    CustomTimes : 1DArray[Int]
        Array of all the time values for use in the Custom option
    ScaleFactor : Int
        Number of iterations to perform the growth model for Exp and Lin
    InitialSample : Int
        Start point to calculate power / s
    """
    arrFilePaths = []
    arrFileName = []
    FolderPath = FolderPath + "/*.csv"
    for filepath in (glob.glob(FolderPath)):
        arrFilePaths.append(filepath)
        arrFileName.append(os.path.splitext(os.path.basename(filepath))[0])
    
    i = 0
    if PrintRawData == True:
        print("Data presented as time scale/s, power for that scale")
    
    while i< len(arrFilePaths):
        
        time, power = PowerProfileCalculator(arrFilePaths[i], ScaleFactor = ScaleFactor, InitialSample = InitialSample, output = True, Visualize = True, GrowthModel = GrowthModel, CustomTimes = CustomTimes, PlotName = arrFileName[i], leg = True)
        if PrintRawData == True:
            print(arrFileName[i])
            j = 0
            while j < len(time):
                print(time[j], power[j])
                j = j+1
        i = i+1
    i = 0

    #PowerProfileCalculator(DataPath, ScaleFactor=10, InitialSample = 5, output = False, Visualize = True, GrowthModel = "Exp", PlotName = "nan")

def PerformanceEstimator(DataPath, AvePowerInput, MaxPowerInput, CDA, RiderMass, BikeMass, TempCorrection = False, Temp = 0):
    """
    WIP
    
    """
    plt.rcParams.update(plt.rcParamsDefault)
    RawData = ReadData(DataPath)


    velocity = 0
    velocityPlotting = [velocity]
    TimeElapsed = [0]
    Mass = RiderMass + BikeMass

    DistanceTraveled = 0

    while DistanceTraveled < (RawData["distance"].iloc[-1]-RawData["distance"].iloc[0]):
        i = RawData['distance'].sub(DistanceTraveled).abs().idxmin()
        Altitude = RawData["altitude"][i]
        if TempCorrection == True:
            tempfix = (Temp+273.15) - fluids.ATMOSPHERE_1976(Altitude).T

            Pressure = fluids.ATMOSPHERE_1976(Altitude, tempfix).rho

        else:
            Pressure = fluids.ATMOSPHERE_1976(Altitude).rho


        WattsDrag = (CDA*(Pressure*velocity*velocity*velocity))/(2)
        AltitudeChange = RawData["altitude"][i]-RawData["altitude"][i+1]
        GPE = Mass*9.8*AltitudeChange

        NetEnergy = AvePowerInput - WattsDrag - GPE
        
        if velocity - ((-2*NetEnergy)/(Mass))**(1/2) <= 0:
            NetEnergy = MaxPowerInput - WattsDrag - GPE

        if NetEnergy > 0: 
            velocity = velocity + ((2*NetEnergy)/(Mass))**(1/2)
        elif NetEnergy < 0:
            velocity = velocity - ((-2*NetEnergy)/(Mass))**(1/2)
        else:
            velocity = velocity
        velocityPlotting.append(velocity)
        TimeElapsed.append(i)
        
        DistanceTraveled = DistanceTraveled + velocity
        i = i + 1
    #print(TimeElapsed,velocityPlotting)
    TimeTaken = TimeElapsed[-1]
    DistanceRidden = (RawData["distance"].iloc[-1]-RawData["distance"].iloc[0])
    DistanceRidden = round(DistanceRidden, 3)
    AveSpeed = mean(velocityPlotting)
    AveSpeed = round(AveSpeed, 3)

    print("Took:", TimeTaken, "s  To travel:",  DistanceRidden,"m  With a average speed of:", AveSpeed, "m/s")
    plt.plot(TimeElapsed,velocityPlotting)


def RiderPowerEstimator(DataPath, RiderMass, BikeMass, CDA, MechanicalDrag, TimeScale = 10, TempCorrection = False):
    """
    WIP
    
    """
    m = RiderMass + BikeMass
    RawData = ReadData(DataPath)
    PowerOut = []
    Time = []
    i = TimeScale
    while i < len(RawData["time"]):
        Velocity=RawData["velocity_smooth"][i]
        VelocityBef=RawData["velocity_smooth"][i-TimeScale]
        AltitudeChange=RawData["altitude"][i]-RawData["altitude"][i-TimeScale]
        Altitude=RawData["altitude"][i-TimeScale:i].mean()
        RiderMass=RiderMass
        BikeMass=BikeMass
        #print("Vel", Velocity,"VelocityBef", VelocityBef,"Wat", Watts,"AltChange", AltitudeChange,"Alt", Altitude,"Mass", RiderMass, BikeMass)
        if TempCorrection == True:
            Temp=RawData["temp"]
            gpe = m*9.8*AltitudeChange

            KEa = 0.5*m*Velocity*Velocity
            KEb = 0.5*m*VelocityBef*VelocityBef

            KE = KEa - KEb
            
            tempfix = (Temp+273.15) - fluids.ATMOSPHERE_1976(Altitude).T

            Pressure = fluids.ATMOSPHERE_1976(Altitude, tempfix).rho

            Drag = (CDA*(Pressure*Velocity*Velocity*Velocity))/(2)

            RiderOutput = Drag + MechanicalDrag + gpe + KE

            PowerOut.append(RiderOutput)
        else:
            gpe = m*9.8*AltitudeChange

            KEa = 0.5*m*Velocity*Velocity
            KEb = 0.5*m*VelocityBef*VelocityBef

            KE = KEa - KEb
            
            Pressure = fluids.ATMOSPHERE_1976(Altitude).rho

            Drag = (CDA*(Pressure*Velocity*Velocity*Velocity))/(2)

            RiderOutput = Drag + MechanicalDrag + gpe + KE
            PowerOut.append(RiderOutput)

        Time.append(RawData["time"][i])
        i = i + TimeScale
    plt.plot(Time, PowerOut)

def TimeCalculator(Distance, StartAltitude, EndAltitude, EnergyInput, RiderMass = 75, BikeMass = 7, CDA = 0.2, Drafting = False, DraftingEffect = 0.6, MechanicalDrag=40, Temp = 0.0, TempFix = False, EnergyPrint = False):
    """
    The Equation Solver needs work
    Parameters
    ----------
    Distance : Float
        Distance of the climb in meters
    StartAltitude : Float
        Altitude at start of climb in meters
    EndAltitude : Float
        Altitude at end of climb in meters
    TimeTaken : Int
        Time taken to climb in seconds
    RiderMass : Float
        Rider mass in kg
    BikeMass : Float
        Bike mass in kg
    MechanicalDrag : Float
        Mechanical drag in watts
    CDA : Float
        CDA of rider bike system
    Drafting : Bool
        If true drops CDA to 0.6 of CDA
    DraftingEffect: Float
        % decrees on the drag
    Temp : Float
        Temperature of day in degrees celsius
    TempFix : Bool
        Toggles using the temp above to calculate air dencity corrected for temp
    ----------
    """
    AltitudeChange = EndAltitude - StartAltitude
    m = RiderMass + BikeMass
    GPE = m*9.8*AltitudeChange

    Altitude = StartAltitude + AltitudeChange/2

    if TempFix == False:
        if Drafting == False:
            Alpha = GPE
            Pressure = fluids.ATMOSPHERE_1976(Altitude).rho
            Beta = (CDA*Pressure*Distance**3)/2
            Gamma = MechanicalDrag
            Rho = EnergyInput
            Time = np.linspace(0,5000, 5)
            temp = EnergyEquation(Time, Alpha, Beta, Gamma, Rho)
            plt.plot(Time,temp)
            data = (Alpha, Beta, Gamma, Rho)
            print(fsolve(MidFunc, 2155, args=data), EnergyEquation(fsolve(MidFunc, 2000, args=data), Alpha, Beta, Gamma, Rho))
        else:
            Pressure = fluids.ATMOSPHERE_1976(Altitude).rho
            TimeTaken = ((0.5*CDA*DraftingEffect*Pressure*Distance**3)/(EnergyInput - (m*9.8*AltitudeChange)))**(1/2)
    else:
        if Drafting == False:
            tempfix = (Temp+273.15) - fluids.ATMOSPHERE_1976(Altitude).T
            Pressure = fluids.ATMOSPHERE_1976(Altitude, tempfix).rho
            TimeTaken = ((0.5*CDA*Pressure*Distance**3)/(EnergyInput - (m*9.8*AltitudeChange)))**(1/2)
        else:
            tempfix = (Temp+273.15) - fluids.ATMOSPHERE_1976(Altitude).T
            Pressure = fluids.ATMOSPHERE_1976(Altitude, tempfix).rho
            TimeTaken = ((0.5*CDA*DraftingEffect*Pressure*Distance**3)/(EnergyInput - (m*9.8*AltitudeChange)))**(1/2)

def MidFunc(x, *data):
    """
    Internal Function used in equation solving
    """
    Alpha, Beta, Gamma, Rho = data
    return(EnergyEquation(x, Alpha, Beta, Gamma, Rho))

def EnergyEquation(Time, Alpha, Beta, Gamma, Rho):
    """
    Internal Function used in equation solving
    """
    return((Alpha-Rho)*Time**2 + Gamma*Time**3 + Beta)


def PerformanceCalculator(Distance, StartAltitude, EndAltitude, TimeTaken, RiderMass = 75, BikeMass = 7, MechanicalDrag = 50, CDA = 0.2, Drafting = False, DraftingEffect = 0.6, Temp = 0.0, TempFix = False, EnergyPrint = False):
    """
    Calculates the average watts and the watts per kilo, a number of these variables are hard to know so play around with things like Mechanical Drag to get a range of rough values, Drafting drops the CDA by DraftingEffect, by defult its set to 0.6 which is around following a bike at around 50-60cm according to: https://doi.org/10.1016/j.proeng.2016.06.186 although I am not convinced by the paper it seems like a rough good guess
    Parameters
    ----------
    Distance : Float
        Distance of the climb in meters
    StartAltitude : Float
        Altitude at start of climb in meters
    EndAltitude : Float
        Altitude at end of climb in meters
    TimeTaken : Int
        Time taken to climb in seconds
    RiderMass : Float
        Rider mass in kg
    BikeMass : Float
        Bike mass in kg
    MechanicalDrag : Float
        Mechanical drag in watts
    CDA : Float
        CDA of rider bike system
    Drafting : Bool
        If true drops CDA to 0.6 of CDA
    DraftingEffect: Float
        % decrees on the drag
    Temp : Float
        Temperature of day in degrees celsius
    TempFix : Bool
        Toggles using the temp above to calculate air dencity corrected for temp
    ----------
    """
    AltitudeChange = EndAltitude - StartAltitude
    m = RiderMass + BikeMass
    GPE = m*9.8*AltitudeChange

    Altitude = StartAltitude + AltitudeChange/2

    Velocity = Distance/TimeTaken
    if TempFix == False:
        if Drafting == False:
            Pressure = fluids.ATMOSPHERE_1976(Altitude).rho
            Drag = (CDA*(Pressure*Velocity*Velocity*Velocity))/(2)
        else:
            Pressure = fluids.ATMOSPHERE_1976(Altitude).rho
            Drag = ((CDA*DraftingEffect)*(Pressure*Velocity*Velocity*Velocity))/(2)
    else:
        if Drafting == False:
            tempfix = (Temp+273.15) - fluids.ATMOSPHERE_1976(Altitude).T
            Pressure = fluids.ATMOSPHERE_1976(Altitude, tempfix).rho
            Drag = (CDA*(Pressure*Velocity*Velocity*Velocity))/(2)
        else:
            tempfix = (Temp+273.15) - fluids.ATMOSPHERE_1976(Altitude).T
            Pressure = fluids.ATMOSPHERE_1976(Altitude, tempfix).rho
            Drag = ((CDA*DraftingEffect)*(Pressure*Velocity*Velocity*Velocity))/(2)
    DragEnergy = Drag*TimeTaken

    MechanicalDragEnergy = MechanicalDrag*TimeTaken
    
    TotalEnergyLost = GPE + DragEnergy + MechanicalDragEnergy
    RiderPowerOutput = TotalEnergyLost/TimeTaken
    WPK = RiderPowerOutput/RiderMass

    RiderPowerOutput = round(float(RiderPowerOutput), 3)
    WPK = round(float(WPK), 3)
    Velocity = round(float(Velocity), 3)

    print("AveWatts: ", RiderPowerOutput, "w , At:", WPK, "w/kg,  For:", TimeTaken, "s,  At a speed of:", Velocity, "m/s")

    if EnergyPrint == True:
        TotalEnergyLost = round(float(TotalEnergyLost), 3)
        GPE = round(float(GPE), 3)
        DragEnergy = round(float(DragEnergy), 3)
        MechanicalDragEnergy = round(float(MechanicalDragEnergy), 3)
        
        print("Rider expended:", TotalEnergyLost, "J,  Gravitational Potential Energy:", GPE, "J,  Aerodynamic Drag Energy:", DragEnergy, "J,  Mechanical Drag Energy:", MechanicalDragEnergy, "J")

def WPKFinderEstimator(DataPath, RiderMass, PrintData=False, GradeHillStart = 1, LengthOfClimbLowerBound = 10, ClimeLengthEnd = 10, NumberProcess = 10, DisplayFitLine = True):
    """
    Function to find the Watts per Kg of a bike rider over a large period of time identifying periods of continuous grade over a threshold value, this function uses a estimate of the WPK derived from VAM from the equation found on https://en.wikipedia.org/wiki/VAM_(bicycling), it seems to underestimate vs raw data approaches probably due to a lack of considering aero and mechanical drag effects. I think this is probably most useful for comparison in the race and also as a relative measure of how hard the race went up each climb. Hill definitions can cause downhills to overpower a uphill creating a anti hill, play around with the settings provided to try and mitigate this.

    Parameters
    ----------
    DataPath : String
        File path of data(Note data must include the grade_smooth value in source which is not automatically ticked)
    RiderMass : Float
        mass of rider in kg, PrintData: if true prints raw values of each data point and annotates plot to allow for each point to be matched
    GradeHillStart : Float
        Threshold value of which defines a hill over it data is recorded under it the hill is decided over(note this may cause climbs with periods of changing gradients to become multiple climbs)
    LengthOfClimbLowerBound : Integer
        If the length of the climb is below this value in seconds it is removed from the data
    ClimeLengthEnd : Integer
        Code sums this number of next gradients, if that sum is smaller than 0 the clime is declared over
    NumberProcess : Integer
        sets the number of performances to use in the plotting. selects the furthest from the origin to the closest. will display all if larger than number of climbs
    DisplayFitLine : Boolean
        toggles the display of a line of best fit defined by a second order polyfit (note not allways desirable to be on as it causes strange fits for low amounts of data or large spreads in data)
    ----------
    """
    plt.rcParams.update(plt.rcParamsDefault)

    RawData = ReadData(DataPath)

    i=0
    AccentTime = [[]]
    AccentAlt = [[]]
    Grade = [[]]

    AccentTimeTemp = []
    AccentAltTemp = []
    GradeTemp = []
    
    while i < len(RawData)-10:
        if(RawData["grade_smooth"][i])>=GradeHillStart:
            AccentTimeTemp.append(RawData["time"][i])
            AccentAltTemp.append(RawData["altitude"][i])
            GradeTemp.append(RawData["grade_smooth"][i])

        elif(sum(RawData["grade_smooth"][i:i+ClimeLengthEnd])<0.0):
            AccentTime.append(AccentTimeTemp)
            AccentAlt.append(AccentAltTemp)
            Grade.append(GradeTemp)

            AccentTimeTemp = []
            AccentAltTemp = []
            GradeTemp = []

        else:
            AccentTimeTemp.append(RawData["time"][i])
            AccentAltTemp.append(RawData["altitude"][i])
            GradeTemp.append(RawData["grade_smooth"][i])
        i = i + 1

    AccentTime.append(AccentTimeTemp)
    AccentAlt.append(AccentAltTemp)
    Grade.append(GradeTemp)

    i = 0
    while i != len(AccentTime):
        if len(AccentTime[i]) <= LengthOfClimbLowerBound:
            AccentTime[i] = []
            AccentAlt[i] = []
            Grade[i] = []
        i = i + 1
    i = 0
    temp = filter(lambda c: c != [], AccentTime)
    AccentTime = list(temp)
    temp = filter(lambda c: c != [], AccentAlt)
    AccentAlt = list(temp)
    temp = filter(lambda c: c != [], Grade)
    Grade = list(temp)

    index = []
    aveWattsPK=[]
    aveGrade=[]
    ElapsedTime = []
    j = 0
    while i != len(AccentTime):
        index.append(i)
        TimeTaken = float(AccentTime[i][-1]-AccentTime[i][0])
        VAM = ((AccentAlt[i][-1]-AccentAlt[i][0])/TimeTaken)*60*60
        EstWattsPK = VAM / (200 + 10*mean(Grade[i]))
        aveWattsPK.append(EstWattsPK)
        aveGrade.append(mean(Grade[i]))
        ElapsedTime.append(AccentTime[i][-1]-AccentTime[i][0])
        i = i + 1
    i = 0


    GoodnessIndex = []
    tempgood = 0.0
    while i != len(AccentTime):
        tempgood = (aveWattsPK[i]/max(aveWattsPK))+(ElapsedTime[i]/max(ElapsedTime))
        GoodnessIndex.append(tempgood)
        i = i + 1
    i = 0

    Rank = sp.argsort(GoodnessIndex)[::-1]
    ProssessingWPK = []
    ProssessingTime = []
    plt.scatter(ElapsedTime,aveWattsPK, c="lightgray")


    if NumberProcess > len(Rank):
        NumberProcess = len(Rank)

    while i < NumberProcess:
        if PrintData == True:
            plt.scatter(ElapsedTime[Rank[i]],aveWattsPK[Rank[i]], c="dimgray", label = ("Starts at:",AccentTime[Rank[i]][0]))
        else:
            plt.scatter(ElapsedTime[Rank[i]],aveWattsPK[Rank[i]], c="k")
        tempname = str(AccentTime[Rank[i]][0])
        #print(tempname)
        tempy = float(aveWattsPK[Rank[i]])
        tempx = float(ElapsedTime[Rank[i]])
        #print(tempx,tempy)
        ProssessingWPK.append(aveWattsPK[Rank[i]])
        ProssessingTime.append(ElapsedTime[Rank[i]])
        if PrintData == True:
            plt.annotate(i, (tempx, tempy))
            grade = mean(Grade[Rank[i]])
            tempname = round(float(tempname), 3)
            VAM = ((AccentAlt[Rank[i]][-1]-AccentAlt[Rank[i]][0])/tempx)*60*60
            tempy = round(float(tempy), 3)
            tempx = round(float(tempx), 3)
            grade = round(grade, 3)
            VAM = round(VAM, 3)
            Start = datetime.timedelta(seconds = tempname)
            Duration = datetime.timedelta(seconds = tempx)

            print("Index:",i,"  Climb Starts at:", Start, "  WPK:", tempy, "w/kg  For:", Duration,"  On a average grade of:", grade, "%  VAM:", VAM, "m/h", "Gain:", (AccentAlt[Rank[i]][-1]-AccentAlt[Rank[i]][0]),"m")
        i = i + 1

    i = 0
    temp = np.polyfit(ProssessingTime,ProssessingWPK,2)
    linefitfunc = np.poly1d(temp)

    linefitx=np.linspace(min(ElapsedTime),max(ElapsedTime),500)
    linefity=linefitfunc(linefitx)
    if DisplayFitLine == True:
        plt.plot(linefitx,linefity, ls = '--', c="gray")
    
    
    plt.xlabel("Time/s")
    plt.ylabel("Watts per Kilo / w kg^-1")
    plt.xlim(left=0)

def WPKFinder(DataPath, RiderMass, PrintData=False, GradeHillStart = 1, LengthOfClimbLowerBound = 10, ClimeLengthEnd = 10, NumberProcess = 10, DisplayFitLine = True):
    """
    Function to find the Watts per Kg of a bike rider over a large period of time identifying periods of continuous grade over a threshhold value, disagreements with strava segment data has been found so take both with a pinch of salt, both are only as good as the altitude data. I think this is probably most usefull for comparason in the race and also as a relative measure of how hard the race went up each climb. Hill definitions can cause downhills to overpower a uphill creating a anti hill, play around with the settings provided to try and mitigate this.

    Parameters
    ----------
    DataPath : String
        File path of data(Note data must include the grade_smooth value in source which is not automatically ticked)
    RiderMass : Float
        mass of rider in kg, PrintData: if true prints raw values of each data point and annotates plot to allow for each point to be matched
    GradeHillStart : Float
        Threshold value of which defines a hill over it data is recorded under it the hill is decided over(note this may cause climbs with periods of changing gradients to become multiple climbs)
    LengthOfClimbLowerBound : Integer
        If the length of the climb is below this value in seconds it is removed from the data
    ClimeLengthEnd : Integer
        Code sums this number of next gradients, if that sum is smaller than 0 the clime is declared over
    NumberProcess : Integer
        sets the number of performances to use in the plotting. selects the furthest from the origin to the closest. will display all if larger than number of climbs
    DisplayFitLine : Boolean
        toggles the display of a line of best fit defined by a second order polyfit (note not allways desirable to be on as it causes strange fits for low amounts of data or large spreads in data)
    ----------
    """
    plt.rcParams.update(plt.rcParamsDefault)

    RawData = ReadData(DataPath)

    i=0
    AccentTime = [[]]
    AccentAlt = [[]]
    Watts = [[]]
    Grade = [[]]

    AccentTimeTemp = []
    AccentAltTemp = []
    WattsTemp = []
    GradeTemp = []
    
    while i < len(RawData)-10:
        if(RawData["grade_smooth"][i])>=GradeHillStart:
            AccentTimeTemp.append(RawData["time"][i])
            AccentAltTemp.append(RawData["altitude"][i])
            WattsTemp.append(RawData["watts"][i])
            GradeTemp.append(RawData["grade_smooth"][i])

        elif(sum(RawData["grade_smooth"][i:i+ClimeLengthEnd])<0.0):
            AccentTime.append(AccentTimeTemp)
            AccentAlt.append(AccentAltTemp)
            Watts.append(WattsTemp)
            Grade.append(GradeTemp)

            AccentTimeTemp = []
            AccentAltTemp = []
            WattsTemp = []
            GradeTemp = []

        else:
            AccentTimeTemp.append(RawData["time"][i])
            AccentAltTemp.append(RawData["altitude"][i])
            WattsTemp.append(RawData["watts"][i])
            GradeTemp.append(RawData["grade_smooth"][i])
        i = i + 1

    AccentTime.append(AccentTimeTemp)
    AccentAlt.append(AccentAltTemp)
    Watts.append(WattsTemp)
    Grade.append(GradeTemp)

    i = 0
    while i != len(AccentTime):
        if len(AccentTime[i]) <= LengthOfClimbLowerBound:
            AccentTime[i] = []
            AccentAlt[i] = []
            Watts[i] = []
            Grade[i] = []
            #print("FileOmmitted")
        i = i + 1
        #print("File Check Loop")
    i = 0
    temp = filter(lambda c: c != [], AccentTime)
    AccentTime = list(temp)
    temp = filter(lambda c: c != [], AccentAlt)
    AccentAlt = list(temp)
    temp = filter(lambda c: c != [], Watts)
    Watts = list(temp)
    temp = filter(lambda c: c != [], Grade)
    Grade = list(temp)

    index = []
    aveWattsPK=[]
    aveGrade=[]
    ElapsedTime = []
    j = 0
    while i != len(AccentTime):
        index.append(i)
        aveWattsPK.append(mean(Watts[i])/RiderMass)
        aveGrade.append(mean(Grade[i]))
        ElapsedTime.append(AccentTime[i][-1]-AccentTime[i][0])
        i = i + 1
    i = 0


    GoodnessIndex = []
    tempgood = 0.0
    while i != len(AccentTime):
        tempgood = (aveWattsPK[i]/max(aveWattsPK))+(ElapsedTime[i]/max(ElapsedTime))
        GoodnessIndex.append(tempgood)
        i = i + 1
    i = 0
    #print(GoodnessIndex)

    Rank = sp.argsort(GoodnessIndex)[::-1]
    ProssessingWPK = []
    ProssessingTime = []
    plt.scatter(ElapsedTime,aveWattsPK, c="lightgray")


    if NumberProcess > len(Rank):
        NumberProcess = len(Rank)

    #print(AccentTime[Rank[i]][0])
    while i < NumberProcess:
        if PrintData == True:
            plt.scatter(ElapsedTime[Rank[i]],aveWattsPK[Rank[i]], c="dimgray", label = ("Starts at:",AccentTime[Rank[i]][0]))
        else:
            plt.scatter(ElapsedTime[Rank[i]],aveWattsPK[Rank[i]], c="k")
        tempname = str(AccentTime[Rank[i]][0])
        #print(tempname)
        tempy = float(aveWattsPK[Rank[i]])
        tempx = float(ElapsedTime[Rank[i]])
        #print(tempx,tempy)
        ProssessingWPK.append(aveWattsPK[Rank[i]])
        ProssessingTime.append(ElapsedTime[Rank[i]])
        if PrintData == True:
            plt.annotate(i, (tempx, tempy))
            grade = mean(Grade[Rank[i]])
            tempname = round(float(tempname), 3)
            VAM = ((AccentAlt[Rank[i]][-1]-AccentAlt[Rank[i]][0])/tempx)*60*60
            tempy = round(float(tempy), 3)
            tempx = round(float(tempx), 3)
            grade = round(grade, 3)
            VAM = round(VAM, 3)
            Start = datetime.timedelta(seconds = tempname)
            Duration = datetime.timedelta(seconds = tempx)

            print("Index:",i,"  Climb Starts at:", Start, "  WPK:", tempy, "w/kg  For:", Duration,"  On a average grade of:", grade, "%  VAM:", VAM, "m/h", "Gain:", (AccentAlt[Rank[i]][-1]-AccentAlt[Rank[i]][0]),"m")
        i = i + 1
    #print(ProssessingWPK,ProssessingTime)
    i = 0
    temp = np.polyfit(ProssessingTime,ProssessingWPK,2)
    linefitfunc = np.poly1d(temp)

    linefitx=np.linspace(min(ElapsedTime),max(ElapsedTime),500)
    linefity=linefitfunc(linefitx)
    if DisplayFitLine == True:
        plt.plot(linefitx,linefity, ls = '--', c="gray")
    
    
    #plt.legend()
    plt.xlabel("Time/s")
    plt.ylabel("Watts per Kilo / w kg^-1")
    plt.xlim(left=0)

"""
Failed exponential fit for above
    popt, pcov = sp.optimize.curve_fit(fitEXPDecay, ProssessingTime, ProssessingWPK)
    
    linefitx=np.linspace(min(ProssessingTime),max(ProssessingTime),500)
    linefity=fitEXPDecay(linefitx, *popt)

    plt.plot(linefitx,linefity, ls = '--', c="gray")

    def fitEXPDecay()

    def fitEXPDecay(x, m, t, b):
        return m * np.exp(-t * x) + b

"""

def calcCDA(Velocity, VelocityBef, Watts, AltitudeChange, Altitude, RiderMass, BikeMass, TimeScale = 1, Temp = 0, TempCorrection = False):
    """
    Internal function that calculates the CDA of a bike rider for CDAPlot

    Parameters
    ----------
    Velocity : Float
        Speed of bike rider, 
    VelocityBef : Float
        Speed of bike rider at the position of Velocity - TimeScale, Watts: Power output of the bike rider, 
    AltitudeChange : Float
        Change in altitude the bike rider has gone through through the timescale, 
    Altitude : Float
        Current altitude of the bike rider, RiderMass: Mass of the bike rider, 
    BikeMass : Float
        Mass of the bike riders bike, 
    TimeScale : Integer
        Length of iterations in seconds 
    Temp : Float
        Passes the temperature recorded, used if TempCorrection = True
    TempCorrection : Boolean
        Toggles a temp correction, if on the data file needs a temp field found in the toggles of Sauce

    Returns
    ----------
    cda : Float
        the calculated CDA
    """
    Riderm = RiderMass
    Bikem = BikeMass
    m = Riderm+Bikem

    gpe = m*9.8*AltitudeChange

    KEa = 0.5*m*Velocity*Velocity
    KEb = 0.5*m*VelocityBef*VelocityBef

    KE = KEa - KEb
    InputEnergy = Watts

    MissingEnergy = InputEnergy - gpe - KE
    
    WattsDrag = MissingEnergy/TimeScale
    

    if TempCorrection == True:
        tempfix = (Temp+273.15) - fluids.ATMOSPHERE_1976(Altitude).T

        Pressure = fluids.ATMOSPHERE_1976(Altitude, tempfix).rho

    else:
        Pressure = fluids.ATMOSPHERE_1976(Altitude).rho


    cda = (2 * (WattsDrag/Velocity))/(Pressure*Velocity*Velocity)
    #print("CDA: ", cda,"Energys ",MissingEnergy, "In",InputEnergy, "GPE",gpe,"EK", KE)

    return cda

def CDAPlot(DataPath, RiderMass, BikeMass, TimeScale = 10, TempCorrection = False, WattsPerCDA = False):
    """
    Plots CDA over time, resolution is TimeScale, data is presented as a box and whisker plot to identify average value, and plot of CDA/m^2 over time/s Function does not contain logic for non aero drag so be advised it is a overestimation. Can create negative values, i think this is from the smoothing in the velocity data available from Strava but that's unconfirmed. Extream high values are caused by events like breaking, bike changes, ect.

    Parameters
    ----------
    DataPath : String
        File path of raw data
    RiderMass : Float
        Mass of bike rider
    BikeMass : Float
        Mass of bike used
    TimeScale : Integer
        Length of iterations across time in seconds
    TempCorrection : Boolean
        Toggles a temp correction, if on the data file needs a temp field found in the toggles of Sauce
    WattsPerCDA : Boolean
        Changes the output to a plot of watts per CDA a usefuel metric for identifying the tradeoff between drag and power output ability
    """
    sns.set(rc={'figure.figsize':(15, 5)})


    RawData = ReadData(DataPath)

    CDAData = []
    Time = []

    i = TimeScale
    while i <= len(RawData["time"])-1:
        Velocity=RawData["velocity_smooth"][i]
        Watts=RawData["watts"][i-TimeScale:i].sum()
        VelocityBef=RawData["velocity_smooth"][i-TimeScale]
        AltitudeChange=RawData["altitude"][i]-RawData["altitude"][i-TimeScale]
        Altitude=RawData["altitude"][i-TimeScale:i].mean()
        RiderMass=RiderMass
        BikeMass=BikeMass
        #print("Vel", Velocity,"VelocityBef", VelocityBef,"Wat", Watts,"AltChange", AltitudeChange,"Alt", Altitude,"Mass", RiderMass, BikeMass)
        if WattsPerCDA == False:
            if TempCorrection == True:
                temp = RawData["temp"][i]
                CDAData.append(calcCDA(Velocity, VelocityBef,  Watts, AltitudeChange, Altitude, RiderMass, BikeMass, TimeScale = TimeScale, Temp = temp, TempCorrection = TempCorrection))
            else:
                CDAData.append(calcCDA(Velocity, VelocityBef,  Watts, AltitudeChange, Altitude, RiderMass, BikeMass, TimeScale = TimeScale))
        else:
            if TempCorrection == True:
                temp = RawData["temp"][i]
                tempory = (Watts/calcCDA(Velocity, VelocityBef,  Watts, AltitudeChange, Altitude, RiderMass, BikeMass, TimeScale = TimeScale, Temp = temp, TempCorrection = TempCorrection))
                #print(tempory)
                CDAData.append(tempory)
            else:
                tempory = (Watts/TimeScale) / calcCDA(Velocity, VelocityBef,  Watts, AltitudeChange, Altitude, RiderMass, BikeMass, TimeScale = TimeScale)
                #print(tempory)
                CDAData.append(tempory)
        Time.append(RawData["time"][i])
        i = i + TimeScale
    #print(CDAData)
    #print(Time)


    data = {'CDA/m^2': [CDAData],
            'Time/s': [Time]}

    PlotFrame = pd.DataFrame(data)


    with sns.axes_style("whitegrid"):
        fig, axs = plt.subplots(nrows=2, constrained_layout=True)

    if WattsPerCDA == False:
        ax = sns.boxplot(x = CDAData,ax=axs[0], width=0.3, color="lightgray")
        add_median_labels(ax)
        #ax.set_xlim(0, 0.7)
        ax.set_xlabel("CDA / m^2")
        ax = sns.lineplot(x = Time, y=CDAData,ax=axs[1],  color="lightgray")
        #ax.set_ylim(0, 0.7)
        ax.set_xlim(min(RawData["time"]),max(RawData["time"]))
        ax.set_ylabel("CDA / m^2")
        ax.set_xlabel("Time / s")
    else:
        ax = sns.boxplot(x = CDAData,ax=axs[0], width=0.3, color="lightgray")
        add_median_labels(ax)
        #ax.set_xlim(0, 0.7)
        ax.set_xlabel("Watts Per CDA / w m-1^-2")
        ax = sns.lineplot(x = Time, y=CDAData,ax=axs[1],  color="lightgray")
        #ax.set_ylim(0, 0.7)
        ax.set_xlim(min(RawData["time"]),max(RawData["time"]))
        ax.set_ylabel("Watts Per CDA / w m-1^-2")
        ax.set_xlabel("Time/s")


def calcDrag(Velocity, VelocityBef, Watts, AltitudeChange, RiderMass, BikeMass, TimeScale = 1):
    """
    Internal function that calculates the drag of a bike rider for CDAPlot
    Parameters
    ----------
    Velocity : Float
        Speed of bike rider
    VelocityBef : Float
        Speed of bike rider at the position of Velocity - TimeScale
    Watts : Float
        Power output of the bike rider
    AltitudeChange : Float
        Change in altitude the bike rider has gone through through the timescale
    RiderMass : Float
        Mass of the bike rider
    BikeMass : Float
        Mass of the bike riders bike
    TimeScale : Integer
        Length of iterations in seconds 

    Returns
    ----------
    Drag : Float
        the calculated drag
    """
    Riderm = RiderMass
    Bikem = BikeMass
    m = Riderm+Bikem

    gpe = m*9.8*AltitudeChange

    KEa = 0.5*m*Velocity*Velocity
    KEb = 0.5*m*VelocityBef*VelocityBef

    KE = KEa - KEb
    InputEnergy = Watts

    MissingEnergy = InputEnergy - gpe - KE
    
    WattsDrag = MissingEnergy/TimeScale

    return WattsDrag

def DragPlot(DataPath, RiderMass, BikeMass, TimeScale = 1):
    """
    Plots drag over time, resolution is TimeScale, data is presented as a drag/w over time/s plot and a drag over speed plot with a function line for interpolation
    
    Parameters
    ----------
    DataPath : String
        File path of raw data
    RiderMass : Float
        Mass of bike rider
    BikeMass : Float
        Mass of bike used
    TimeScale : Integer
        Length of iterations across time in seconds
    """
    sns.set(rc={'figure.figsize':(15, 5)})


    RawData = ReadData(DataPath)

    Drag = []
    Time = []
    Velocitys = []

    i = TimeScale
    while i <= len(RawData["time"])-1:
        Velocity=RawData["velocity_smooth"][i]
        Watts=RawData["watts"][i-TimeScale:i].sum()
        VelocityBef=RawData["velocity_smooth"][i-TimeScale]
        AltitudeChange=RawData["altitude"][i]-RawData["altitude"][i-TimeScale]
        Altitude=RawData["altitude"][i-TimeScale:i].mean()
        RiderMass=RiderMass
        BikeMass=BikeMass
        #print("Vel", Velocity,"VelocityBef", VelocityBef,"Wat", Watts,"AltChange", AltitudeChange,"Alt", Altitude,"Mass", RiderMass, BikeMass)
        Drag.append(calcDrag(Velocity, VelocityBef,  Watts, AltitudeChange, RiderMass, BikeMass, TimeScale = TimeScale))
        Time.append(float(RawData["time"][i]))
        Velocitys.append(Velocity)
        i = i + TimeScale
    #print(CDAData)
    #print(Time)

    data = {'CDA/m^2': [Drag],
            'Time/s': [Time]}

    PlotFrame = pd.DataFrame(data)


    with sns.axes_style("whitegrid"):
        fig, axs = plt.subplots(nrows=2, constrained_layout=True)

    ax = sns.lineplot(x = Time, y=Drag,ax=axs[0])
    ax.set_ylabel("Drag/w")
    ax.set_xlabel("Time/s")

    p = sp.polyfit(Velocitys,Drag,2)
    # mymodel = sp.poly1d(sp.polyfit(Velocitys,Drag,2))

    # myline = sp.linspace(0, 16, 1)

    # print(p)
    # print(mymodel)

    ax = sns.regplot(x = Velocitys, y=Drag,ax=axs[1], order = 2, line_kws={'label':"y={0:.3f}x^2+{1:.3f}x+{2:.3f}".format(p[0],p[1],p[2])})

    ax.set_ylabel("Drag/w")
    #ax.set_xlim(0,16)
    #ax.set_ylim(0,600)

    ax.set_xlabel("Velocity/ms^-1")
    ax.legend()

def add_median_labels(ax, precision='.3f'):
    """
    function to add median data to box and whisker plots
    solution adapted from Christian Karcher's answer to:
    https://stackoverflow.com/questions/38649501/labeling-boxplot-in-seaborn-with-median-value

    Parameters
    ----------
    ax : Matplotlib Axis
        axis to add the label
    precision : String
        number of decimal places to add to plot
    """
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{precision}}', ha='center', va='center',
                       fontweight='bold', color='white')
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])
