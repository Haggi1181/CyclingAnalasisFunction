

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