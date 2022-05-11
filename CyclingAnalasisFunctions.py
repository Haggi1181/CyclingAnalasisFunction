import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.patheffects as path_effects
import fluids
from statistics import mean
import datetime

plt.close("all")

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199

sns.set(rc={'figure.figsize':(15, 5)})


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

def WPKFinder(DataPath, RiderMass, PrintData=False, GradeHillStart = 1, LengthOfclimbLowerBound = 10, NumberProcess = 10, DisplayFitLine = True):
    """
    Function to find the Watts per Kg of a bike rider over a large period of time identifying periods of continuous grade over a threshhold value, dissagreements with strava segment data has been found so take both with a pinch of salt, both are only as good as the altitude data. I think this is probably most usefull for comparason in the race and also as a relative mesure of how hard the race went up each climb. Hill definitions can cause downhills to overpower a uphill creating a anti hill, play around with the settings provided to try and mitigate this.

    Parameters
    ----------
    DataPath : String
        File path of data(Note data must include the grade_smooth value in source which is not automatically ticked)
    RiderMass : Float
        mass of rider in kg, PrintData: if true prints raw values of each data point and annotates plot to allow for each point to be matched
    GradeHillStart : Float
        Threshold value of which defines a hill over it data is recorded under it the hill is decided over(note this may cause climbs with periods of changing gradients to become multiple climbs)
    LengthOfclimbLowerBound : Integer
        if the length of the climb is below this value in seconds it is removed from the data
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


        elif(RawData["grade_smooth"][i+1])<0 and (RawData["grade_smooth"][i+2])<0 and (RawData["grade_smooth"][i+3])<0 and (RawData["grade_smooth"][i+4])<0 and (RawData["grade_smooth"][i+5])<0 and (RawData["grade_smooth"][i+6])<0 and (RawData["grade_smooth"][i+7])<0 and (RawData["grade_smooth"][i+8])<0 and (RawData["grade_smooth"][i+9])<0 and (RawData["grade_smooth"][i+10])<0:#lol this is the worst thing ive ever made i did a r/badprogramming meme
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
        if len(AccentTime[i]) <= LengthOfclimbLowerBound:
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

def calcCDA(Velocity, VelocityBef, Watts, AltitudeChange, Altitude, RiderMass, BikeMass, TimeScale = 1):
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
    

    Pressure = fluids.ATMOSPHERE_1976(Altitude).rho
    cda = (2 * (WattsDrag/Velocity))/(Pressure*Velocity*Velocity)
    #print("CDA: ", cda,"Energys ",MissingEnergy, "In",InputEnergy, "GPE",gpe,"EK", KE)

    return cda

def CDAPlot(DataPath, RiderMass, BikeMass, TimeScale = 10):
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
    """
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



        CDAData.append(calcCDA(Velocity, VelocityBef,  Watts, AltitudeChange, Altitude, RiderMass, BikeMass, TimeScale = TimeScale))
        Time.append(RawData["time"][i])
        i = i + TimeScale
    #print(CDAData)
    #print(Time)

    data = {'CDA/m^2': [CDAData],
            'Time/s': [Time]}

    PlotFrame = pd.DataFrame(data)


    with sns.axes_style("whitegrid"):
        fig, axs = plt.subplots(nrows=2, constrained_layout=True)

    ax = sns.boxplot(x = CDAData,ax=axs[0], width=0.3, color="lightgray")
    add_median_labels(ax)
    ax.set_xlim(0, 0.7)
    ax.set_xlabel("CDA/m^2")
    ax = sns.lineplot(x = Time, y=CDAData,ax=axs[1],  color="lightgray")
    #ax.set_ylim(0, 0.7)
    ax.set_xlim(min(RawData["time"]),max(RawData["time"]))
    ax.set_ylabel("CDA/m^2")
    ax.set_xlabel("Time/s")


def calcDrag(Velocity, VelocityBef, Watts, AltitudeChange, Altitude, RiderMass, BikeMass, TimeScale = 1):
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
    Altitude : Float
        Current altitude of the bike rider
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
        Drag.append(calcDrag(Velocity, VelocityBef,  Watts, AltitudeChange, Altitude, RiderMass, BikeMass, TimeScale = TimeScale))
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