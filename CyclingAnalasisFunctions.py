import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde
from scipy import stats
import numpy as np
import glob
import os
import seaborn as sns
import matplotlib.patheffects as path_effects
import fluids

plt.close("all")

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199

sns.set(rc={'figure.figsize':(15, 5)})


def ReadData(DataPath):
    Data = pd.read_csv(DataPath).fillna(0)

    #fixes the missing values for watts
    Data.columns = Data.columns.str.lstrip()
    Data["watts"] = Data["watts"].replace("      ", int(0))
    Data["watts"] = pd.to_numeric(Data["watts"])

    return Data

def calcCDA(Velocity, VelocityBef, Watts, AltitudeChange, Altitude, RiderMass, BikeMass, TimeScale = 1):
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

def CDAPlot(DataPath, RiderMass, BikeMass, TimeScale = 1):
    RawData = ReadData(DataPath)

    CDAData = []
    Time = []

    i = TimeScale
    while i <= len(RawData["time"])-1:
        Velocity=RawData["velocity_smooth"][i]
        Watts=RawData["watts"][i-TimeScale:i].sum()
        VelocityBef=RawData["velocity_smooth"][i-TimeScale]
        AltitudeChange=RawData["altitude"][i]-RawData["altitude"][i-TimeScale]
        Altitude=RawData["altitude"][i]
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
        Altitude=RawData["altitude"][i]
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




def CalcTeamFromInv(DataPath, TeamCodeHeader = "Team Code", PointsHeader = "Points"):
    raw = pd.read_csv(DataPath)
    mid = raw.groupby(TeamCodeHeader).head(10).sort_values(by=[PointsHeader],ascending=[False])
    fin = mid.groupby(TeamCodeHeader).sum().sort_values(by=[PointsHeader],ascending=[False])
    fin = fin.drop("Rank", axis = 1)
    return fin

def PlotCalTeamFromInv(DataPath, NumberOfTeams = 30, TeamCodeHeader = "Team Code", PointsHeader = "Points"):
    colors = []
    colors[:18] = ["b"]*18
    colors[19:NumberOfTeams] = ["r"]*(NumberOfTeams - 19)
    #print(colors)
    CalcTeamFromInv(DataPath, TeamCodeHeader = TeamCodeHeader, PointsHeader = PointsHeader)[0:NumberOfTeams].plot.bar(xlabel = "Team Code", ylabel = "Points", legend=False, color = colors)

def VelocityWatsDencityPloter(x, y, s=1):
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    #plt.xlim(5,20)
    #plt.ylim(0,1000)
    #plt.colorbar(z)
    plt.xlabel("Speed / m/s")
    plt.ylabel("Power / w")
    plt.scatter(x, y, c=z, s=s)

def VelocityWatsDencity(FilePath,s = 1):
    Data = pd.read_csv(FilePath).fillna(0)
    VelocityWatsDencityPloter(x = Data[" velocity_smooth"],y = Data[" watts"], s = s)

def DragCalculateFolder(FolderPath, MassFile, debug = False):
    arrFilePaths = []
    arrFileName = []
    FolderPath = FolderPath + "/*.csv"
    for filepath in (glob.glob(FolderPath)):
        arrFilePaths.append(filepath)
        arrFileName.append(os.path.splitext(os.path.basename(filepath))[0])
        #print("File Loaded")


    i=0
    while i != len(arrFilePaths):
        if arrFilePaths[i] == MassFile:
            arrFilePaths[i] = "asd"
            arrFileName[i] = "asd"
            #print("FileOmmitted")
        i = i + 1
        #print("File Check Loop")

    temp = filter(lambda c: c != "asd", arrFilePaths)
    arrFilePaths = list(temp)
    temp = filter(lambda c: c != "asd", arrFileName)
    arrFileName = list(temp)
    #print("Files Trimmed")
    i=0

    RiderData = pd.read_csv(MassFile)

    while i< len(arrFilePaths):
        
        temprow = RiderData.loc[RiderData['RiderName'] == arrFileName[i]]
        print("Rider: ", arrFileName[i], "   Bike Brand: ", str(temprow["BikeBrand"]))
        print(temprow)
        ridermass = float(temprow["RiderMass"])
        bikemass = float(temprow["BikeMass"])


        print("Rider: ", arrFileName[i], "   Bike Brand: ", str(temprow["BikeBrand"]))
        DragCalculator(arrFilePaths[i], ridermass, bikemass)
        i = i+1
        #print("Data Read")
    i = 0






def DragCalculator(DataPath, RiderMass, BikeMass, debug = False):

    Testing = pd.read_csv(DataPath).fillna(0)

    #fixes the missing values for watts
    Testing.columns = Testing.columns.str.lstrip()
    Testing["watts"] = Testing["watts"].replace("      ", int(0))
    Testing["watts"] = pd.to_numeric(Testing["watts"])
    #Drop in solution contained between comments

    if debug == True:
        print("Min Alt: ",min(Testing["altitude"]), "m,  Max Alt: ", max(Testing["altitude"]), "m,  Diff: ", max(Testing["altitude"]) - min(Testing["altitude"]), "m")

    deltaAlt = max(Testing["altitude"]) - min(Testing["altitude"])

    Riderm = RiderMass
    Bikem = BikeMass
    m = Riderm+Bikem

    gpe = m*9.8* deltaAlt

    KEi = 0.5*m*Testing["velocity_smooth"].iat[0]*Testing["velocity_smooth"].iat[0]
    KEf = 0.5*m*Testing["velocity_smooth"].iat[-1]*Testing["velocity_smooth"].iat[-1]

    KE = KEf - KEi


    InputEnergy = sum(Testing["watts"])

    MissingEnergy = InputEnergy - gpe - KE
    if debug == True:
        print("GPE:", gpe,", Kinetic Energy Change: ", KE, ", Input Energy: ",InputEnergy, "j", ", Missing Energy: ", MissingEnergy, "j")

    
    ElapsedTime = Testing["time"].iat[-1] - Testing["time"].iat[0]
    WattsDrag = MissingEnergy/ElapsedTime
    
    aveSpeed = Testing["velocity_smooth"].mean()
    aveSpeedkph = aveSpeed*3.6
    DragDevidedByaveSpeed = WattsDrag / aveSpeed
    DragDevidedByaveSpeedSqur = WattsDrag / (aveSpeed**2)
    DragDevidedByaveSpeedSqurkph = WattsDrag / ((aveSpeed*3.6)**2)

    if debug == True:
        print("Ave Speed:", aveSpeed, ", Power Per Unit Velocity: ", DragDevidedByaveSpeed, ", Power Per Unit Velocity Squared: ", DragDevidedByaveSpeedSqur)

    WattsDrag = round(WattsDrag, 2)
    DragDevidedByaveSpeedSqur = round(DragDevidedByaveSpeedSqur, 2)
    DragDevidedByaveSpeedSqurkph = round(DragDevidedByaveSpeedSqurkph, 2)
    aveSpeed = round(aveSpeed, 2)
    aveSpeedkph = round(aveSpeedkph, 2)

    print("Watt Drag Est: ", WattsDrag, "w", ", Power Per Unit Velocity Squared: ", DragDevidedByaveSpeedSqur, "ws^4/m^2", ", At Speed: ", aveSpeed, "m/s")
    print("Watt Drag Est: ", WattsDrag, "w", ", Power Per Unit Velocity Squared: ", DragDevidedByaveSpeedSqurkph, "wh^4/k^2", ", At Speed: ", aveSpeedkph, "k/h")


def CDACalc(DataPath, RiderMass, BikeMass, debug = False, output = False):

    Testing = pd.read_csv(DataPath).fillna(0)

    #fixes the missing values for watts
    Testing.columns = Testing.columns.str.lstrip()
    Testing["watts"] = Testing["watts"].replace("      ", int(0))
    Testing["watts"] = pd.to_numeric(Testing["watts"])
    #Drop in solution contained between comments

    if debug == True:
        print("Min Alt: ",min(Testing["altitude"]), "m,  Max Alt: ", max(Testing["altitude"]), "m,  Diff: ", max(Testing["altitude"]) - min(Testing["altitude"]), "m")

    deltaAlt = max(Testing["altitude"]) - min(Testing["altitude"])

    Riderm = RiderMass
    Bikem = BikeMass
    m = Riderm+Bikem

    gpe = m*9.8* deltaAlt

    KEi = 0.5*m*Testing["velocity_smooth"].iat[0]*Testing["velocity_smooth"].iat[0]
    KEf = 0.5*m*Testing["velocity_smooth"].iat[-1]*Testing["velocity_smooth"].iat[-1]

    KE = KEf - KEi


    InputEnergy = sum(Testing["watts"])

    MissingEnergy = InputEnergy - gpe - KE
    if debug == True:
        print("GPE:", gpe,", Kinetic Energy Change: ", KE, ", Input Energy: ",InputEnergy, "j", ", Missing Energy: ", MissingEnergy, "j")

    
    ElapsedTime = Testing["time"].iat[-1] - Testing["time"].iat[0]
    WattsDrag = MissingEnergy/ElapsedTime
    
    aveSpeed = Testing["velocity_smooth"].mean()
    aveSpeedkph = aveSpeed*3.6
    DragDevidedByaveSpeed = WattsDrag / aveSpeed
    DragDevidedByaveSpeedSqur = WattsDrag / (aveSpeed**2)
    DragDevidedByaveSpeedSqurkph = WattsDrag / ((aveSpeed*3.6)**2)

    if debug == True:
        print("Ave Speed:", aveSpeed, ", Power Per Unit Velocity: ", DragDevidedByaveSpeed, ", Power Per Unit Velocity Squared: ", DragDevidedByaveSpeedSqur)

    WattsDrag = round(WattsDrag, 2)
    DragDevidedByaveSpeedSqur = round(DragDevidedByaveSpeedSqur, 2)
    DragDevidedByaveSpeedSqurkph = round(DragDevidedByaveSpeedSqurkph, 2)
    aveSpeed = round(aveSpeed, 2)
    aveSpeedkph = round(aveSpeedkph, 2)

    if debug == True:
        print("Watt Drag Est: ", WattsDrag, "w", ", Power Per Unit Velocity Squared: ", DragDevidedByaveSpeedSqur, "ws^4/m^2", ", At Speed: ", aveSpeed, "m/s")
        print("Watt Drag Est: ", WattsDrag, "w", ", Power Per Unit Velocity Squared: ", DragDevidedByaveSpeedSqurkph, "wh^4/k^2", ", At Speed: ", aveSpeedkph, "k/h")

    avePressure = ((fluids.ATMOSPHERE_1976(min(Testing["altitude"])).rho + fluids.ATMOSPHERE_1976(max(Testing["altitude"])).rho)/2)
    cda = (2 * (WattsDrag/aveSpeed))/(avePressure*aveSpeed*aveSpeed)
    if output == True:
        return cda
    else:
        print("CDA Est:",cda, "m^2")


def CDAComparason(Rider1DataPath, Rider1Mass, Rider1BikeMass, Rider2DataPath, Rider2Mass, Rider2BikeMass, debug = False):
    CDA1 = CDACalc(Rider1DataPath, Rider1Mass, Rider1BikeMass, output = True)
    CDA2 = CDACalc(Rider2DataPath, Rider2Mass, Rider2BikeMass, output = True)

    deltaCDA = CDA1 - CDA2
    
    drag1,cdaish1,avespe1 = DragCalculatorReturnData(Rider1DataPath, Rider1Mass, Rider1BikeMass)
    
    Testing = pd.read_csv(Rider1DataPath).fillna(0)
    Testing.columns = Testing.columns.str.lstrip()
    Testing["watts"] = Testing["watts"].replace("      ", int(0))
    Testing["watts"] = pd.to_numeric(Testing["watts"])
    avePressure = fluids.ATMOSPHERE_1976(min(Testing["altitude"])).rho+((fluids.ATMOSPHERE_1976(min(Testing["altitude"])).rho - fluids.ATMOSPHERE_1976(max(Testing["altitude"])).rho)/2)
    

    DeltaDrag = (1/2)*(avespe1)*(avespe1)*(avespe1)*deltaCDA*avePressure
    #print(DeltaDrag)

    NewDrag = drag1-DeltaDrag

    EK = avespe1*avespe1*0.5*(Rider1Mass+Rider1BikeMass) + NewDrag
    NewV = (2*EK/((Rider1Mass+Rider1BikeMass)))**(1/2)
    #print(NewV)
    changeinSpead = NewV - avespe1

    elapstDist = Testing["distance"].iat[-1] - Testing["distance"].iat[0]
    elapstTime = Testing["time"].iat[-1] - Testing["time"].iat[0]

    newTime = elapstDist/NewV

    changeintime = elapstTime - newTime


    changeintime = round(changeintime, 3)
    changeinSpead = round(changeinSpead, 3)
    DeltaDrag = round(DeltaDrag, 3)

    print("Rider 1 would have gone: ",changeintime,"s Faster, with a increase in speed of: ", changeinSpead, "m/s, and a change in drag of", DeltaDrag, "w")



def DragCalculatorReturnData(DataPath, RiderMass, BikeMass, debug = False):

    Testing = pd.read_csv(DataPath).fillna(0)

    #fixes the missing values for watts
    Testing.columns = Testing.columns.str.lstrip()
    Testing["watts"] = Testing["watts"].replace("      ", int(0))
    Testing["watts"] = pd.to_numeric(Testing["watts"])
    #Drop in solution contained between comments

    if debug == True:
        print("Min Alt: ",min(Testing["altitude"]), "m,  Max Alt: ", max(Testing["altitude"]), "m,  Diff: ", max(Testing["altitude"]) - min(Testing["altitude"]), "m")

    deltaAlt = max(Testing["altitude"]) - min(Testing["altitude"])

    Riderm = RiderMass
    Bikem = BikeMass
    m = Riderm+Bikem

    gpe = m*9.8* deltaAlt

    KEi = 0.5*m*Testing["velocity_smooth"].iat[0]*Testing["velocity_smooth"].iat[0]
    KEf = 0.5*m*Testing["velocity_smooth"].iat[-1]*Testing["velocity_smooth"].iat[-1]

    KE = KEf - KEi


    InputEnergy = sum(Testing["watts"])

    MissingEnergy = InputEnergy - gpe - KE
    if debug == True:
        print("GPE:", gpe,", Kinetic Energy Change: ", KE, ", Input Energy: ",InputEnergy, "j", ", Missing Energy: ", MissingEnergy, "j")

    
    ElapsedTime = Testing["time"].iat[-1] - Testing["time"].iat[0]
    WattsDrag = MissingEnergy/ElapsedTime
    
    aveSpeed = Testing["velocity_smooth"].mean()
    aveSpeedkph = aveSpeed*3.6
    DragDevidedByaveSpeed = WattsDrag / aveSpeed
    DragDevidedByaveSpeedSqur = WattsDrag / (aveSpeed**2)
    DragDevidedByaveSpeedSqurkph = WattsDrag / ((aveSpeed*3.6)**2)

    if debug == True:
        print("Ave Speed:", aveSpeed, ", Power Per Unit Velocity: ", DragDevidedByaveSpeed, ", Power Per Unit Velocity Squared: ", DragDevidedByaveSpeedSqur)

    WattsDrag = round(WattsDrag, 2)
    DragDevidedByaveSpeedSqur = round(DragDevidedByaveSpeedSqur, 2)
    DragDevidedByaveSpeedSqurkph = round(DragDevidedByaveSpeedSqurkph, 2)
    aveSpeed = round(aveSpeed, 2)
    aveSpeedkph = round(aveSpeedkph, 2)

    return(WattsDrag, DragDevidedByaveSpeedSqur, aveSpeed)

def DragCalculateFolderPlotterAvrages(FolderPath, MassFile, debug = False):
    arrFilePaths = []
    arrFileName = []
    FolderPath = FolderPath + "/*.csv"
    for filepath in (glob.glob(FolderPath)):
        arrFilePaths.append(filepath)
        arrFileName.append(os.path.splitext(os.path.basename(filepath))[0])
        #print("File Loaded")


    i=0
    while i != len(arrFilePaths):
        if arrFilePaths[i] == MassFile:
            arrFilePaths[i] = "asd"
            arrFileName[i] = "asd"
            #print("FileOmmitted")
        i = i + 1
        #print("File Check Loop")

    temp = filter(lambda c: c != "asd", arrFilePaths)
    arrFilePaths = list(temp)
    temp = filter(lambda c: c != "asd", arrFileName)
    arrFileName = list(temp)
    #print("Files Trimmed")
    i=0

    RiderData = pd.read_csv(MassFile)


    PlotData = [[]]
    while i< len(arrFilePaths):
        temprow = RiderData.loc[RiderData['RiderName'] == arrFileName[i]]
        if debug == True:
            print("Rider: ", arrFileName[i], "   Bike Brand: ", str(temprow["BikeBrand"]))
            print(temprow)
        ridermass = float(temprow["RiderMass"])
        bikemass = float(temprow["BikeMass"])

        if debug == True:
            print("Rider: ", arrFileName[i], "   Bike Brand: ", str(temprow["BikeBrand"]))

        drag,cdaish,avespe = DragCalculatorReturnData(arrFilePaths[i], ridermass, bikemass)
        #plt.scatter(cdaish, ridermass)

        PlotData.append([arrFileName[i], temprow["BikeBrand"].values[0], ridermass, bikemass, cdaish, drag, avespe])



        i = i+1
        #print("Data Read")
    #print(PlotData)
    PlotFrame = pd.DataFrame(PlotData)
    PlotFrame.columns =['RiderName', 'BikeBrand', "RiderMass/kg", "BikeMass/kg", "DragDivSpeed^2/ws^4m^-2", "Drag/w", "AveSpeed/ms^-2"]
    PlotFrame = PlotFrame.iloc[1: , :]
    #if debug == True:
    #print(PlotFrame)
    sns.set(rc={'figure.figsize':(15, 5)})
    #sns.scatterplot(data=PlotFrame, x='Drag/Speed^2', y='RiderMass', hue='BikeBrand', )
    standard_deviations = 1
    #print(type(PlotFrame["Drag/Speed^2"][1]))
    #sns.scatterplot(data=PlotFrame, x='Drag/Speed^2', y='RiderMass', hue='BikeBrand', )
    with sns.axes_style("whitegrid"):
        fig, axs = plt.subplots(nrows=3, constrained_layout=True)

    ax = sns.boxplot(x=PlotFrame['Drag/w'],ax=axs[0], width=0.3, color="lightgray")
    add_median_labels(ax)
    ax = sns.boxplot(x=PlotFrame['RiderMass/kg'],ax=axs[1], width=0.3, color="lightgray")
    add_median_labels(ax)
    ax = sns.boxplot(x=PlotFrame['DragDivSpeed^2/ws^4m^-2'],ax=axs[2], width=0.3, color="lightgray")
    add_median_labels(ax)

    i = 0


def DragCalculateFolderPlotterScatters(FolderPath, MassFile, debug = False):
    arrFilePaths = []
    arrFileName = []
    FolderPath = FolderPath + "/*.csv"
    for filepath in (glob.glob(FolderPath)):
        arrFilePaths.append(filepath)
        arrFileName.append(os.path.splitext(os.path.basename(filepath))[0])
        #print("File Loaded")


    i=0
    while i != len(arrFilePaths):
        if arrFilePaths[i] == MassFile:
            arrFilePaths[i] = "asd"
            arrFileName[i] = "asd"
            #print("FileOmmitted")
        i = i + 1
        #print("File Check Loop")

    temp = filter(lambda c: c != "asd", arrFilePaths)
    arrFilePaths = list(temp)
    temp = filter(lambda c: c != "asd", arrFileName)
    arrFileName = list(temp)
    #print("Files Trimmed")
    i=0

    RiderData = pd.read_csv(MassFile)


    PlotData = [[]]
    while i< len(arrFilePaths):
        temprow = RiderData.loc[RiderData['RiderName'] == arrFileName[i]]
        if debug == True:
            print("Rider: ", arrFileName[i], "   Bike Brand: ", str(temprow["BikeBrand"]))
            print(temprow)
        ridermass = float(temprow["RiderMass"])
        bikemass = float(temprow["BikeMass"])

        if debug == True:
            print("Rider: ", arrFileName[i], "   Bike Brand: ", str(temprow["BikeBrand"]))

        drag,cdaish,avespe = DragCalculatorReturnData(arrFilePaths[i], ridermass, bikemass)
        #plt.scatter(cdaish, ridermass)

        PlotData.append([arrFileName[i], temprow["BikeBrand"].values[0], ridermass, bikemass, cdaish, drag, avespe])



        i = i+1
        #print("Data Read")
    #print(PlotData)
    PlotFrame = pd.DataFrame(PlotData)
    PlotFrame.columns =['RiderName', 'BikeBrand', "RiderMass/kg", "BikeMass/kg", "DragDivSpeed^2/ws^4m^-2", "Drag/w", "AveSpeed/ms^-2"]
    PlotFrame = PlotFrame.iloc[1: , :]
    #if debug == True:
    #print(PlotFrame)
    sns.set(rc={'figure.figsize':(20, 5)})
    #sns.scatterplot(data=PlotFrame, x='Drag/Speed^2', y='RiderMass', hue='BikeBrand', )
    standard_deviations = 1
    #print(type(PlotFrame["Drag/Speed^2"][1]))
    #sns.scatterplot(data=PlotFrame, x='Drag/Speed^2', y='RiderMass', hue='BikeBrand', )
    with sns.axes_style("whitegrid"):
        fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True)

    ax = sns.regplot(x=PlotFrame['RiderMass/kg'], y = PlotFrame['Drag/w'],ax=axs[0][0])
    
    ax = sns.regplot(x=PlotFrame['RiderMass/kg'], y = PlotFrame['DragDivSpeed^2/ws^4m^-2'],ax=axs[0][1])

    ax = sns.regplot(x=PlotFrame['AveSpeed/ms^-2'], y = PlotFrame['Drag/w'],ax=axs[1][0])

    ax = sns.regplot(x=PlotFrame['AveSpeed/ms^-2'], y = PlotFrame['DragDivSpeed^2/ws^4m^-2'],ax=axs[1][1])


    i = 0


def add_median_labels(ax, precision='.3f'):
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